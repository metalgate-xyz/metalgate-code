"""BaseSandbox / SandboxProvider implementation backed by macOS sandbox-exec.

Registered under the `deepagents_code.sandbox_providers` entry point (see
pyproject.toml) -- installing this package is enough for
`dcode --sandbox seatbelt` to work, no config.toml edits required.

Requirements: macOS. `sandbox-exec` ships with the OS (no install needed).
It is deprecated by Apple with no official replacement, but remains
functional and receives security fixes on current macOS releases.

Security model: there is no separate workspace directory. The launch
directory -- the project ``dcode`` was started from -- is the read/write
area. The Seatbelt SBPL profile is the only fence: reads are allowed
broadly across the host (required for process bootstrap) but **denied
under ``/Users``**, then re-allowed for the launch directory and any
``read_paths`` (configured via
``[sandboxes.providers.seatbelt.params]`` in ``~/.deepagents/config.toml``).
Writes are confined to the launch directory (plus ``/dev/null`` for shell
redirections). System files outside ``/Users`` stay readable (Apple's,
reinstallable). Network is **allowed by default** (coding work needs it:
pip, npm, git, tests, dev servers); pass ``network=False`` to opt out.
Every operation -- ``execute``, ``upload_files``, ``download_files`` --
runs under ``sandbox-exec``, so the profile is the single, uniform fence.
There is no manual path-containment check: the kernel-level profile
enforces it, and there is no virtual root to translate because the model
uses the real launch-dir paths it is told.

Seatbelt has no mount namespaces (unlike bubblewrap), so there is no
PID/mount isolation and no bind-mount of a virtual ``/workspace``. Treat
this as protection against accidental scope creep, not a hard boundary
against adversarial code.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import concurrent.futures
import functools
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from deepagents.backends.protocol import (
    ASYNC_GREP_TIMEOUT,
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    GrepResult,
)
from deepagents.backends.sandbox import (
    BaseSandbox,
    _parse_grep_output,
)

try:
    from deepagents_code._env_vars import SERVER_ENV_PREFIX
    from deepagents_code.integrations.sandbox_provider import (
        SandboxProvider,
        SandboxProviderMetadata,
    )
    from deepagents_code.project_utils import get_server_project_context
except ImportError as exc:  # pragma: no cover
    msg = (
        "dcode-seatbelt-sandbox requires deepagents-code to be installed. "
        "Install with: pip install 'dcode-seatbelt-sandbox[dcode]'"
    )
    raise ImportError(msg) from exc

_DEFAULT_TIMEOUT = 60

_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]+\Z")
"""Permitted sandbox_id characters. The id flows only into a profile
filename under ``_PROFILES_ROOT``, so this charset exists to keep that
filename safe (no path separators, no whitespace, no shell metacharacters).
It is a friendly-error fast-fail, not a security boundary: there is no
per-sandbox directory to escape."""

_PROFILES_ROOT = Path("/tmp") / "dcode-seatbelt-profiles"
"""Seatbelt SBPL profiles are written here, outside any path the sandboxed
process can read, so the confined process can't read the profile that
confines it."""


def _offload_if_running_loop[T](fn: Callable[[], T]) -> T:
    """Run a blocking callable off the event loop when one is running.

    dcode builds the server graph on a langgraph event loop that guards
    against synchronous blocking calls (the `blockbuster` detector rejects
    `os.access`, `os.getcwd`, `os.mkdir`, `os.listdir`, `os.stat`, ...). The
    graph-build path calls `SandboxProvider.get_or_create` synchronously on
    that loop, so the filesystem calls it makes (`shutil.which` -> os.access,
    `Path.resolve`) would trip the guard.

    When a running loop is present, this runs `fn` in a worker thread (which
    has no running loop, so the guard allows blocking I/O there) and blocks on
    the result. With no running loop (pytest, sync CLI callers), `fn` runs
    inline. The same `asyncio.to_thread` mechanism is what dcode's own
    `SandboxProvider.aget_or_create` uses for the async path.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return fn()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(functools.partial(fn)).result()


def _launch_dir_raw() -> str:
    """Return the launch dir as a raw string, with no filesystem calls.

    Safe to call from `metadata` during registry discovery (which runs on
    the langgraph-guarded event loop): it only reads environment variables
    (plain dict lookups, not watched by blockbuster) and never resolves
    symlinks. The canonical (resolved) path is computed in `get_or_create`,
    which is offloaded to a worker thread. Falls back to ``/tmp`` when no
    server context is present (direct CLI/pytest callers).
    """
    raw = os.environ.get(f"{SERVER_ENV_PREFIX}PROJECT_ROOT") or os.environ.get(
        f"{SERVER_ENV_PREFIX}CWD"
    )
    return raw or "/tmp"


def _resolve_launch_dir() -> Path:
    """Resolve the launch dir to its canonical path (following symlinks).

    Prefers the project root (git root); falls back to the user cwd; falls
    back to the process cwd only when no server context exists (direct
    CLI/pytest callers). The resolved path is what the SBPL `subpath`
    literal must match for kernel I/O (e.g. ``/tmp`` -> ``/private/tmp``).
    """
    ctx = get_server_project_context()
    if ctx is not None:
        base = ctx.project_root if ctx.project_root is not None else ctx.user_cwd
        return base.resolve()
    return Path.cwd().resolve()


def _running_venv() -> Path | None:
    """Return the venv dcode runs in, or None for the system interpreter.

    ``sys.prefix`` is the active venv root; it equals ``sys.base_prefix``
    only outside a venv. Re-allowed under /Users so the inherited PATH
    resolves to its binaries even when dcode is launched outside the venv's
    project.
    """
    if sys.prefix == sys.base_prefix:
        return None
    return Path(sys.prefix).resolve()


def _agent_home() -> Path:
    """Return the dcode profile root (``DEEPAGENTS_HOME`` or the default)."""
    raw = os.environ.get("DEEPAGENTS_HOME")
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / ".deepagents").resolve()


def _agent_rw_paths() -> list[Path]:
    """Return existing agent-authoring paths under the profile root.

    ``agent/AGENTS.md`` (agent memory) and ``global_tools`` (agent repo tool
    source) live under the profile root, outside the launch dir when dcode
    runs against a different project. Re-allowed read-write under /Users.
    ``global_tools`` is created by the dynamic_tools plugin, so it exists
    when dcode starts.
    """
    home = _agent_home()
    candidates = [home / "agent" / "AGENTS.md", home / "global_tools"]
    return [p.resolve() for p in candidates if p.exists()]


_PROFILE_HEADER = """\
(version 1)
(deny default)
(allow process-fork)
(allow process-exec)
(allow signal (target self))
(allow sysctl-read)
;; BSD routing sockets (AF_ROUTE) are gated by seatbelt's `system-socket`
;; operation, which is distinct from `network*` (the latter governs
;; bind/connect/send/recv on a socket, not socket creation). `socket(AF_ROUTE,
;; SOCK_RAW, 0)` is the standard Darwin mechanism for enumerating interfaces
;; and monitoring route changes (e.g. Go's netmon.New() opens one to watch
;; for network state changes); without this allow it fails with EPERM even
;; when network is otherwise allowed. Scoped to AF_ROUTE so other raw/system
;; sockets stay denied. Mirrors Apple's own mDNSResponderHelper.sb.
(allow system-socket
    (socket-domain AF_ROUTE))
;; TLS trust evaluation on Darwin goes through Mach IPC to trustd /
;; SecurityServer, not just the root cert files. Without these allows,
;; HTTPS certificate verification fails and `go mod download` can't fetch.
(allow mach-lookup
    (global-name "com.apple.trustd")
    (global-name "com.apple.trustd.agent")
    (global-name "com.apple.securityd.xpc")
    (global-name "com.apple.SecurityServer")
    (global-name "com.apple.TrustEvaluationAgent")
    (global-name "com.apple.ocspd"))
;; Broad read access is required for a process to bootstrap under
;; (deny default): dyld must read the shared cache, frameworks, locale data,
;; and the dynamic linker needs file-read-data/metadata across /usr, /System,
;; /Library, /opt (Homebrew on Apple Silicon), etc. Reads of /Users are then
;; denied (all three op types -- file-read* alone is not enough; ls/stat go
;; through file-read-data/metadata) and re-allowed only for the launch dir,
;; the curated toolchain/config paths, and any read_paths.
;; SBPL is last-match-wins, so every re-allow rule comes AFTER the deny.
(allow file-read*)
(allow file-read-data)
(allow file-read-metadata)
(deny file-read*
    (subpath "/Users"))
(deny file-read-data
    (subpath "/Users"))
(deny file-read-metadata
    (subpath "/Users"))
;; `cd` resolves each ancestor of the target path with a stat() (metadata).
;; /Users and every launch-dir ancestor under /Users sit under the deny
;; above, so without a metadata-only re-allow on each of them, `cd` into the
;; launch dir fails with "Not a directory". Metadata-only is enough for
;; traversal; listing an ancestor's contents (file-read-data) stays denied.
{ancestor_metadata}
;; Full read access on the launch dir and on each tool/read path under /Users.
;; Paths outside /Users are already readable via the broad allow and are
;; skipped by the builder below.
{read_allows}
;; Writes: launch dir only (plus /dev/null for shell redirections). Default
;; deny already blocks every other write, including all of /Users outside the
;; launch dir -- no separate /Users write-deny is needed.
{write_allows}
(allow file-write*
    (subpath "{launch}"))
(allow file-write*
    (literal "/dev/null"))
;; Go's toolchain (`go list`, `go build`) creates temp work dirs under /tmp
;; (via /tmp, not TMPDIR), so a coding sandbox must allow writing there.
(allow file-write*
    (subpath "/private/tmp"))
"""
"""Static part of the SBPL profile. Placeholders:
- ``{launch}``: the resolved (canonical, non-symlink) launch dir.
- ``{ancestor_metadata}``: metadata-only re-allow rules for ``/Users`` and
  each launch-dir ancestor (for `cd` traversal).
- ``{read_allows}``: full read re-allow rules for the launch dir, the curated
  tool paths, and any ``read_paths`` under ``/Users``. The network rule is
  appended after this header.
- ``{write_allows}``: write re-allow rules for curated tool cache paths under
  ``/Users`` (e.g. the Go build cache) that compilers/LSP servers must write
  to function. The launch-dir write rule follows."""


# Toolchain and config subpaths under $HOME that a coding sandbox needs to
# read. These are re-allowed on top of the /Users deny so compilers,
# runtimes, and VCS keep working without exposing secrets. Mirrors the
# enumeration real-world bubblewrap wrappers use (bind-mount a finite list
# of home subpaths read-only rather than all of $HOME). Each entry is a
# relative path resolved against $HOME; nonexistent ones are skipped at
# profile-build time. Extend via `read_paths` in config.toml for project- or
# user-specific paths.
_TOOL_READ_PATHS = (
    # VCS
    ".gitconfig",
    ".config/git",
    # Python (uv-managed toolchain, pip cache). uv's cache is read while
    # resolving/extracting wheels and building the sdist index.
    ".local/share/uv",
    ".local/bin",
    ".cache/pip",
    ".cache/uv",
    "Library/Caches/pip",
    "Library/Caches/uv",
    # Node
    ".npmrc",
    ".local/lib/node_modules",
    ".cache/corepack",
    # Rust
    ".cargo",
    ".rustup",
    # Go
    ".cache/go-build",
    "Library/Caches/go-build",
    "go",
    "go.work",
    "Library/Caches/gopls",
    # Cargo/rust crates index
    ".cargo/registry",
)


# Toolchain cache subpaths under $HOME that a coding sandbox must WRITE to.
# Compilers and language servers (gopls, rust-analyzer) build metadata into
# these caches; denying writes breaks type checking and go-to-definition even
# though the source is readable. Each entry is a relative path resolved against
# $HOME; nonexistent ones are skipped at profile-build time. Symmetric to
# `_TOOL_READ_PATHS`.
_TOOL_WRITE_PATHS = (
    # Go: gopls writes compiled package metadata to the build cache. Without
    # write access it can't load views, so textDocument/definition returns null.
    "Library/Caches/go-build",
    ".cache/go-build",
    # Go: `go mod download` writes downloaded modules to the module cache at
    # $GOPATH/pkg/mod (default ~/go/pkg/mod). The whole ~/go tree must be
    # writable so Go can create pkg/mod and its cache/sum subdirs from scratch.
    "go",
    # gopls's own cache (typerefs, export data, diagnostics). Without write
    # access gopls logs errors on every request and degrades.
    "Library/Caches/gopls",
    ".cache/gopls",
    # uv: writes downloaded wheels and the sdist git index (sdists-v9/.git)
    # here; a read-only cache makes installs fail with "Operation not
    # permitted" on the .git index.
    ".cache/uv",
    "Library/Caches/uv",
)


def _tool_read_paths() -> list[str]:
    """Resolve `_TOOL_READ_PATHS` against $HOME, returning existing /Users paths.

    Nonexistent paths are silently skipped (a tool not installed shouldn't
    generate noise). Paths outside /Users are skipped too -- they're already
    covered by the broad read allow.
    """
    home = Path.home()
    out: list[str] = []
    for rel in _TOOL_READ_PATHS:
        resolved = str((home / rel).resolve())
        if not resolved.startswith("/Users/"):
            continue
        out.append(resolved)
    return out


def _tool_write_paths() -> list[str]:
    """Resolve `_TOOL_WRITE_PATHS` against $HOME, returning existing /Users paths.

    Same skip rules as `_tool_read_paths`: nonexistent paths are dropped (a
    tool not installed generates no cache to write), and paths outside /Users
    are skipped (writable system paths aren't under the /Users write fence).
    """
    home = Path.home()
    out: list[str] = []
    for rel in _TOOL_WRITE_PATHS:
        resolved = str((home / rel).resolve())
        if not resolved.startswith("/Users/"):
            continue
        out.append(resolved)
    return out


def _ancestor_metadata_rules(launch: str) -> str:
    """Metadata-only re-allow rules for /Users and each launch-dir ancestor.

    `cd /Users/<you>/code/proj` stat's `/Users`, `/Users/<you>`,
    `/Users/<you>/code` before reaching the launch dir. Each sits under the
    `/Users` deny and would block `cd` with "Not a directory" without these
    rules. Only `file-read-metadata` is re-allowed -- enough for path
    resolution, not enough to list an ancestor's contents.
    """
    parts = Path(launch).parts  # ('/', 'Users', '<you>', 'code', 'proj')
    acc = ""
    lines: list[str] = []
    for part in parts[1:-1]:  # skip leading '/', skip the launch dir itself
        acc += "/" + part
        if acc.startswith("/Users"):
            lines.append(f'(allow file-read-metadata (subpath "{acc}"))')
    return "\n".join(lines)


def _read_allow_block(launch: str, read_paths: list[str]) -> str:
    """Build the full-read re-allow rules for /Users subpaths.

    Covers the launch dir itself, the curated tool paths, and any user
    ``read_paths``. Each gets all three read operation types
    (``file-read*``, ``file-read-data``, ``file-read-metadata``) since the
    ``/Users`` deny blocks all three and a partial re-allow would leave
    ``ls``/``stat`` broken. Paths are resolved to canonical form so the
    ``subpath`` literal matches kernel I/O (e.g. ``/tmp`` -> ``/private/tmp``).
    Paths outside ``/Users`` are skipped (already covered by the broad allow).
    """
    all_paths = [launch, *_tool_read_paths(), *read_paths]
    venv = _running_venv()
    if venv is not None:
        all_paths.append(str(venv))
    all_paths.extend(str(p) for p in _agent_rw_paths())
    lines: list[str] = []
    seen: set[str] = set()
    for raw in all_paths:
        resolved = str(Path(raw).expanduser().resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        if not resolved.startswith("/Users/"):
            continue
        lines.append(
            f'(allow file-read* (subpath "{resolved}"))\n'
            f'(allow file-read-data (subpath "{resolved}"))\n'
            f'(allow file-read-metadata (subpath "{resolved}"))'
        )
    return "\n".join(lines)


def _write_allow_block() -> str:
    """Build the write re-allow rules for curated tool cache /Users subpaths.

    Mirrors `_read_allow_block` for the tool caches that compilers and language
    servers must write to (see `_TOOL_WRITE_PATHS`). Paths outside /Users are
    skipped: the /Users write deny is implicit under ``(deny default)`` and
    non-/Users paths are not fenced. Paths are resolved to canonical form so
    the ``subpath`` literal matches kernel I/O (e.g. ``/tmp`` -> ``/private/tmp``).
    """
    lines: list[str] = []
    seen: set[str] = set()
    for raw in [*_tool_write_paths(), *(str(p) for p in _agent_rw_paths())]:
        resolved = str(Path(raw).expanduser().resolve())
        if resolved in seen or not resolved.startswith("/Users/"):
            continue
        seen.add(resolved)
        lines.append(f'(allow file-write* (subpath "{resolved}"))')
    return "\n".join(lines)


def _build_bsd_grep_cmd(
    pattern: str,
    path: str | None,
    glob: str | None,
    max_count: int | None = None,
) -> str:
    """Build a grep command whose output ``_parse_grep_output`` can parse.

    The base ``BaseSandbox.grep`` uses ``grep -rHnFZ``. ``-Z`` is GNU grep's
    ``--null`` (a NUL after the filename), which the parser relies on to split
    ``path\\0line:text`` records unambiguously. On macOS the seatbelt sandbox
    runs BSD grep, where ``-Z`` is ``--decompress`` (zgrep mode) and emits plain
    ``path:line:text`` -- so every match becomes an unparseable line and grep
    returns a hard error instead of results.

    The basename-glob branch mirrors ``_build_grep_cmd`` but uses the portable
    long option ``--null`` in place of ``-Z``. ``--null`` is the
    NUL-after-filename flag on both BSD grep and GNU grep, so the parser works
    unchanged.

    Slash-containing globs can't use ``grep --include`` (basename-only), so
    they run an in-process Python search; see ``_grep_path_glob_cmd_safe``.
    """
    if glob and "/" in glob:
        return _grep_path_glob_cmd_safe(pattern, path, glob, max_count)

    search_path = shlex.quote(path or ".")
    grep_opts = "-rHnF --null"
    pattern_escaped = shlex.quote(pattern)
    glob_pattern = f"--include={shlex.quote(glob)}" if glob else ""
    base = f"grep {grep_opts} {glob_pattern} -e {pattern_escaped} {search_path} 2>/dev/null"
    if max_count is not None:
        return f"{base} | head -n {int(max_count) + 1} || true"
    return f"{base} || true"


def _grep_path_glob_cmd_safe(
    pattern: str,
    path: str | None,
    glob: str,
    max_count: int | None,
) -> str:
    """Slash-glob grep command that survives the seatbelt ``sh -c`` wrapping.

    Resolves a ``/``-containing glob in-process and emits ``path\\0line:text``
    records. The Python script is fed to ``python3 -`` through a
    quoted-delimiter heredoc (``<<'PY'``), which makes the shell treat the
    body as literal text, so the script's ``'``, ``"``, and ``$`` characters
    never reach shell parsing. Search path, glob, pattern, and max_count
    travel as ``sys.argv`` tokens, ``shlex.quote``-d against the inner shell.
    """
    body = (
        "import glob, os, sys\n"
        "search_path, glob_pat, pattern, mc = sys.argv[1:]\n"
        "max_count = int(mc) if mc else None\n"
        "match_count = 0\n"
        "if os.path.isdir(search_path):\n"
        "    os.chdir(search_path)\n"
        "    rel_glob = glob_pat.lstrip('/')\n"
        "    if any(seg == '..' for seg in rel_glob.replace(chr(92), '/').split('/')):\n"
        "        sys.stderr.write('glob contains path traversal\\n')\n"
        "        sys.exit(2)\n"
        "    real_root = os.path.realpath(search_path)\n"
        "    rel_files = sorted(glob.glob(rel_glob, recursive=True))\n"
        "    targets = []\n"
        "    for rel in rel_files:\n"
        "        real_open = os.path.realpath(rel)\n"
        "        if real_open != real_root and not real_open.startswith(real_root + os.sep):\n"
        "            continue\n"
        "        display_path = os.path.join(search_path, os.path.relpath(real_open, real_root))\n"
        "        targets.append((real_open, display_path))\n"
        "else:\n"
        "    targets = [(search_path, search_path)]\n"
        "for open_path, display_path in targets:\n"
        "    try:\n"
        "        with open(open_path, 'r', encoding='utf-8', errors='ignore') as fh:\n"
        "            for i, line in enumerate(fh, 1):\n"
        "                if pattern in line:\n"
        "                    sys.stdout.write(display_path + chr(0) + str(i) + ':' + line.rstrip(chr(10)) + chr(10))\n"
        "                    match_count += 1\n"
        "                    if max_count is not None and match_count > max_count:\n"
        "                        sys.exit(0)\n"
        "    except OSError:\n"
        "        pass\n"
    )
    max_count_str = str(int(max_count)) if max_count is not None else ""
    args = [path or ".", glob, pattern, max_count_str]
    head = "python3 - " + " ".join(shlex.quote(a) for a in args) + " 2>/dev/null <<'PY'"
    return f"{head}\n{body}\nPY"


class SeatbeltSandbox(BaseSandbox):
    """Local sandbox backend that runs every operation under `sandbox-exec`.

    The launch directory is the read/write area. **Reads** are allowed
    broadly across the host -- required for a process to bootstrap under
    ``(deny default)`` (dyld shared cache, frameworks, ``/opt/homebrew``,
    locale data) -- but reads of ``/Users`` are denied, then re-allowed for
    the launch directory and any ``read_paths``. **Writes** are confined to
    the launch directory (plus ``/dev/null``). System files outside
    ``/Users`` remain readable. Network is allowed by default unless
    ``network=False`` was passed at creation. Each command runs with a
    minimal, secret-free environment (PATH, HOME, TMPDIR, SHELL, LANG)
    instead of inheriting dcode's process environment.

    ``upload_files`` and ``download_files`` are routed through ``execute``
    (via base64 over the sandboxed process's stdin/stdout) so the Seatbelt
    profile fences them exactly like ordinary commands -- there are no
    direct host filesystem calls in the provider process.
    """

    def __init__(
        self,
        sandbox_id: str,
        launch_dir: Path,
        *,
        network: bool = True,
        read_paths: list[str] | None = None,
    ) -> None:
        self._id = sandbox_id
        # Resolve to the canonical path now: every SBPL `subpath` literal must
        # match the kernel's view of the path (e.g. /tmp -> /private/tmp), and
        # the profile and write fence silently fail to apply if a caller passes
        # an unresolved path. get_or_create already resolves, but direct
        # construction (used by tests and the validation suite) may not.
        self._launch = launch_dir.resolve()
        self._network = network
        self._read_paths = list(read_paths) if read_paths else []
        # The Seatbelt profile is written at most once per instance, on the
        # first execute()/upload_files()/download_files() call (see
        # _ensure_profile). Constructing a sandbox does not touch the
        # filesystem -- this keeps the validation suite (which probes
        # _profile_text() and the abstract surface without sandbox-exec) free
        # of side effects, and defers the write to when a profile is actually
        # needed.
        self._profile_path: Path | None = None
        self._profile_lock = threading.Lock()

    @property
    def id(self) -> str:
        return self._id

    def _env(self) -> dict[str, str]:
        # Don't inherit dcode's environment: it typically carries API keys and
        # tokens (OPENAI_API_KEY, GITHUB_TOKEN, ...) the sandboxed code must not
        # see or exfiltrate. Pass a minimal, secret-free environment instead.
        # HOME is the real user home, not the launch dir: toolchains resolve
        # their caches relative to HOME (Go: ~/go, ~/Library/Caches/go-build;
        # Rust: ~/.cargo; ...), and the SBPL profile -- not HOME -- is the
        # fence that keeps secrets (~/.ssh, ~/.aws, ...) unreadable.
        #
        # PATH is *inherited* from the host rather than hardcoded. PATH is not
        # a secret -- it is just a list of directories -- and the curated read
        # re-allows (`_tool_read_paths`) make the user's toolchains
        # (~/.cargo/bin, ~/.local/bin, ...) readable so their binaries can exec,
        # but those binaries are unreachable unless their dirs are on PATH. A
        # hardcoded PATH would make the agent blind to the very tools the
        # re-allows were added to expose; inheriting the host PATH lets the
        # agent use the same tools the user does. (PATH entries whose dirs are
        # under /Users but not re-allowed are blocked by the /Users deny -- a
        # `command not found`, not a leak or a hole; entries outside /Users are
        # readable and exec is allowed via `(allow process-exec)`.)
        return {
            "PATH": os.environ.get(
                "PATH", "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
            ),
            "HOME": str(Path.home()),
            # TMPDIR inside the launch dir (a writable area); created lazily by
            # the sandboxed command itself, not by the provider process.
            "TMPDIR": str(self._launch / ".tmp"),
            "SHELL": "/bin/sh",
            "LANG": "en_US.UTF-8",
        }

    def _profile_text(self) -> str:
        launch = str(self._launch)
        read_allows = _read_allow_block(launch, self._read_paths)
        write_allows = _write_allow_block()
        ancestor_metadata = _ancestor_metadata_rules(launch)
        header = _PROFILE_HEADER.format(
            launch=launch,
            read_allows=read_allows,
            write_allows=write_allows,
            ancestor_metadata=ancestor_metadata,
        )
        network_rule = "(allow network*)" if self._network else "(deny network*)"
        return header + network_rule + "\n"

    def _write_profile(self) -> Path:
        """Write the profile to disk (unconditionally) and return its path.

        Validates the sandbox id and writes ``_profile_text()`` to
        ``_PROFILES_ROOT / f"{self._id}.sb"``. This is a forced write; the
        production path (``_argv``) calls ``_ensure_profile`` instead, which
        writes at most once per instance so concurrent ``sandbox-exec -f``
        readers never observe a half-written file (the original code rewrote
        on every call and readers failed with "no version specified"). Direct
        callers (the test suite) use this to materialise a profile on disk,
        which is safe because no reader is running yet.
        """
        _PROFILES_ROOT.mkdir(parents=True, exist_ok=True)
        if not _ID_PATTERN.match(self._id):
            msg = (
                f"Invalid sandbox_id {self._id!r}: must match [A-Za-z0-9._-]+ "
                "(no path separators or whitespace)."
            )
            raise ValueError(msg)
        profile_path = _PROFILES_ROOT / f"{self._id}.sb"
        profile_path.write_text(self._profile_text())
        return profile_path

    def _ensure_profile(self) -> Path:
        """Return the profile path, writing it at most once per instance.

        dcode fans out parallel tool calls on a single sandbox. The profile is
        a pure function of construction-time state, so it is written once --
        under this lock -- and every subsequent call reuses the path. The lock
        serializes the single write so that even under concurrent calls no two
        writers truncate the file at once, and readers (``sandbox-exec -f``)
        only run after this returns, so they see the complete file.
        """
        if self._profile_path is not None:
            return self._profile_path
        with self._profile_lock:
            if self._profile_path is None:
                self._profile_path = self._write_profile()
        return self._profile_path

    def _argv(self, command: str) -> list[str]:
        return [
            "sandbox-exec",
            "-f",
            str(self._ensure_profile()),
            "sh",
            "-c",
            f"cd {shlex.quote(str(self._launch))} && {command}",
        ]

    def _run(
        self,
        argv: list[str],
        *,
        stdin: bytes | None = None,
        timeout: int | None = None,
    ) -> tuple[bytes, bytes, int | None]:
        """Run `argv` under sandbox-exec, returning raw stdout/stderr/exit_code.

        Bytes throughout (no `text=True`) so binary file content round-trips
        exactly through the base64 transport used by upload/download. A
        timeout yields the partial output captured so far and ``None`` exit.
        """
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                timeout=timeout or _DEFAULT_TIMEOUT,
                env=self._env(),
                check=False,
                input=stdin,
            )
        except subprocess.TimeoutExpired as exc:
            return bytes(exc.stdout or b""), bytes(exc.stderr or b""), None
        return proc.stdout, proc.stderr, proc.returncode

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        out, err, rc = self._run(self._argv(command), timeout=timeout)
        output = (out + err).decode("utf-8", errors="replace")
        if rc is None:
            output += f"\n[command timed out after {timeout or _DEFAULT_TIMEOUT}s]"
        return ExecuteResponse(output=output, exit_code=rc)

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
    ) -> GrepResult:
        """Search file contents for a literal string under the seatbelt fence.

        Overrides ``BaseSandbox.grep`` to build the command with
        ``_build_bsd_grep_cmd`` instead of the base ``grep -rHnFZ``: BSD grep
        (the one macOS ``sandbox-exec`` runs) treats ``-Z`` as ``--decompress``,
        not the NUL separator the base parser expects, so the inherited
        command returns a parse error for every match. ``--null`` produces the
        same ``path\\0line:text`` records on BSD grep, so the base
        ``_parse_grep_output`` is reused unchanged.
        """
        result = self.execute(_build_bsd_grep_cmd(pattern, path, glob, max_count))
        return _parse_grep_output(result, path, max_count)

    async def agrep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
    ) -> GrepResult:
        """Async version of `grep`, mirroring the base timeout guard."""
        try:
            result = await asyncio.wait_for(
                self.aexecute(_build_bsd_grep_cmd(pattern, path, glob, max_count)),
                timeout=ASYNC_GREP_TIMEOUT,
            )
        except TimeoutError:
            return GrepResult(
                error=f"Error: grep timed out after {ASYNC_GREP_TIMEOUT}s. "
                "Try a more specific pattern or a narrower path.",
            )
        return _parse_grep_output(result, path, max_count)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        responses: list[FileUploadResponse] = []
        for path, content in files:
            qpath = shlex.quote(path)
            qparent = shlex.quote(str(Path(path).parent))
            # mkdir the parent (no-op if it exists) then decode base64 from
            # stdin into the target. The write fence (target must be inside
            # the launch dir) is enforced by the Seatbelt profile, not here.
            command = f"mkdir -p {qparent} && base64 -d > {qpath}"
            _out, err, rc = self._run(
                self._argv(command),
                stdin=base64.b64encode(content),
            )
            if rc == 0:
                responses.append(FileUploadResponse(path=path, error=None))
            else:
                detail = err.decode("utf-8", errors="replace").strip()
                responses.append(FileUploadResponse(path=path, error=detail or f"exit {rc}"))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        responses: list[FileDownloadResponse] = []
        for path in paths:
            qpath = shlex.quote(path)
            # macOS `base64 -i <file>` reads the file and writes encoded bytes
            # to stdout; GNU base64 accepts a positional path, but macOS does
            # not, so -i is the portable form.
            out, err, rc = self._run(self._argv(f"base64 -i {qpath}"))
            if rc == 0:
                try:
                    content = base64.b64decode(out)
                except (ValueError, binascii.Error):
                    detail = "base64 decode failed"
                    responses.append(FileDownloadResponse(path=path, content=None, error=detail))
                    continue
                responses.append(FileDownloadResponse(path=path, content=content, error=None))
                continue
            detail = err.decode("utf-8", errors="replace").lower()
            if "no such file" in detail or "not found" in detail:
                responses.append(FileDownloadResponse(path=path, content=None, error="file_not_found"))
            else:
                msg = err.decode("utf-8", errors="replace").strip() or f"exit {rc}"
                responses.append(FileDownloadResponse(path=path, content=None, error=msg))
        return responses


class SeatbeltProvider(SandboxProvider):
    """SandboxProvider that creates/deletes local sandbox-exec-isolated sandboxes."""

    @property
    def metadata(self) -> SandboxProviderMetadata:
        # `working_dir` is the path the model is told to operate in. It is read
        # by the registry during discovery (which instantiates this provider on
        # the langgraph-guarded event loop), so it must not make blocking fs
        # calls -- `_launch_dir_raw` only reads env vars. The canonical path
        # is resolved in `get_or_create` (offloaded to a worker thread).
        return SandboxProviderMetadata(
            name="seatbelt",
            working_dir=_launch_dir_raw(),
            supports_sandbox_id=True,
            supports_snapshot_name=False,
            backend_module=None,
        )

    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        network: bool = True,
        read_paths: list[str] | None = None,
        **kwargs: Any,
    ) -> SeatbeltSandbox:
        def _create() -> SeatbeltSandbox:
            if shutil.which("sandbox-exec") is None:
                msg = (
                    "sandbox-exec is not available. This provider only works on "
                    "macOS, where sandbox-exec ships with the OS."
                )
                raise RuntimeError(msg)
            if sandbox_id is not None and not _ID_PATTERN.match(sandbox_id):
                msg = (
                    f"Invalid sandbox_id {sandbox_id!r}: must match "
                    "[A-Za-z0-9._-]+ (no path separators or whitespace)."
                )
                raise ValueError(msg)
            sid = sandbox_id or f"seatbelt-{uuid.uuid4().hex[:12]}"
            launch = _resolve_launch_dir()
            return SeatbeltSandbox(
                sid, launch, network=network, read_paths=read_paths
            )

        # Offload the blocking fs calls (`shutil.which`/`Path.resolve`) when
        # called on the langgraph-guarded event loop at graph build time; run
        # inline for sync callers (pytest, CLI).
        return _offload_if_running_loop(_create)

    def delete(self, *, sandbox_id: str, **kwargs: Any) -> None:
        # No per-sandbox workspace exists -- only the profile file. The id is
        # validated by the charset check before it touches the filesystem.
        if not _ID_PATTERN.match(sandbox_id):
            msg = (
                f"Invalid sandbox_id {sandbox_id!r}: must match [A-Za-z0-9._-]+ "
                "(no path separators or whitespace)."
            )
            raise ValueError(msg)
        (_PROFILES_ROOT / f"{sandbox_id}.sb").unlink(missing_ok=True)
