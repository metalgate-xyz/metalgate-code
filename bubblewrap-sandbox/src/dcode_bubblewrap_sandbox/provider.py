"""BaseSandbox / SandboxProvider implementation backed by Linux bubblewrap (`bwrap`).

Registered under the `deepagents_code.sandbox_providers` entry point (see
pyproject.toml) -- installing this package is enough for
`dcode --sandbox bubblewrap` to work, no config.toml edits required.

Requirements: Linux with `bwrap` installed and unprivileged user namespaces
enabled. Bubblewrap v0.12 dropped the legacy setuid mode, so unprivileged
user namespaces are now mandatory: `bwrap` creates a new user namespace
(`--unshare-user-try`) and uses it to create the mount/pid/ipc/uts/cgroup
namespaces that form the sandbox. If the kernel disallows unprivileged user
namespaces, `bwrap` cannot function and the first command surfaces that.

Security model (mirrors `dcode-seatbelt-sandbox`, adapted to bubblewrap's
namespace-and-bind-mount mechanism): there is no separate workspace directory.
The launch directory -- the project `dcode` was started from -- is the
read/write area, bind-mounted read-write at its real path so the agent
operates on the real launch-dir paths it is told (no virtual root to
translate, just like seatbelt). The sandbox root is an empty tmpfs; system
trees needed for a process to bootstrap (`/usr`, `/etc`, `/bin`, `/lib`,
`/proc`, `/dev`, ...) are bind-mounted **read-only**. The rest of the host
filesystem -- including all of `$HOME` except the launch dir and a curated
list of toolchain/config subpaths -- is simply **not mounted**, so it is
invisible rather than merely denied: `~/.ssh`, `~/.aws`, other users' files,
`/var`, `/run` sockets, etc. are absent. `read_paths` (configured via
``[sandboxes.providers.bubblewrap.params]`` in ``~/.deepagents/config.toml``)
adds extra read-only bind mounts on top of that.

Writes are confined to the launch dir: nothing else is mounted writable, so
every other path is either read-only (system binds) or non-existent
(everything else). Network is **allowed by default** (coding work needs it:
pip, npm, git, tests, dev servers): the host network namespace is shared.
Pass ``network=False`` to opt out, which adds ``--unshare-net`` (loopback
only). Every operation -- ``execute``, ``upload_files``, ``download_files``
-- runs under `bwrap`, so the mount namespace is the single, uniform fence.
There is no manual path-containment check: the kernel-level namespace
enforces it, and there is no virtual root to translate because the model
uses the real launch-dir paths it is told.

Unlike seatbelt, bubblewrap gives real container-style isolation: PID
namespace (the sandbox can't see or signal host processes), IPC namespace,
UTS namespace, and a private mount namespace with an empty root. Combined
with ``--new-session`` (setsid, blocking the TIOCSTI terminal-injection
attack) and ``--die-with-parent`` (the sandbox dies if the provider exits),
this is a stronger boundary than seatbelt's syscall filter -- but it still
should not be treated as a hard boundary against adversarial code with
kernel exploits.
"""

from __future__ import annotations

import base64
import binascii
import concurrent.futures
import functools
import os
import re
import shlex
import shutil
import subprocess
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox

try:
    from deepagents_code._env_vars import SERVER_ENV_PREFIX
    from deepagents_code.integrations.sandbox_provider import (
        SandboxProvider,
        SandboxProviderMetadata,
    )
    from deepagents_code.project_utils import get_server_project_context
except ImportError as exc:  # pragma: no cover
    msg = (
        "dcode-bubblewrap-sandbox requires deepagents-code to be installed. "
        "Install with: pip install 'dcode-bubblewrap-sandbox[dcode]'"
    )
    raise ImportError(msg) from exc

_DEFAULT_TIMEOUT = 60

_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]+\Z")
"""Permitted sandbox_id characters. For bubblewrap the id is not used to name
any on-disk artifact (there is no profile file and no per-sandbox workspace),
so this charset exists purely for API parity with the seatbelt provider and a
friendly-error fast-fail -- not a security boundary."""


# Read-only system binds every bubblewrap sandbox needs so a process can
# bootstrap and the common toolchains keep working. `/usr` and `/etc` are hard
# binds (essential; a missing one means the sandbox can't run anything), the
# rest use `--ro-bind-try` so layouts where `/bin`/`/lib`/`/lib64`/`/sbin` are
# symlinks into `/usr` (merged-/usr distros) or absent don't cause a hard
# failure. Two `/run` subpaths are bound so systemd-resolved's
# `/etc/resolv.conf` symlink target exists (DNS keeps working under the host
# network namespace without exposing the rest of `/run`'s sockets).
_SYSTEM_RO_BINDS: tuple[tuple[str, str, bool], ...] = (
    # (src, dst, try) -- try=True uses --ro-bind-try (skip if missing).
    ("/usr", "/usr", False),
    ("/etc", "/etc", False),
    ("/bin", "/bin", True),
    ("/lib", "/lib", True),
    ("/lib64", "/lib64", True),
    ("/sbin", "/sbin", True),
    ("/opt", "/opt", True),
    # systemd-resolved stub + classic resolvconf: /etc/resolv.conf is often a
    # symlink into one of these. Bind only these subdirs, not all of /run
    # (which holds D-Bus/systemd sockets that could be abused for execution).
    ("/run/systemd/resolve", "/run/systemd/resolve", True),
    ("/run/resolvconf", "/run/resolvconf", True),
)


# Toolchain and config subpaths that a coding sandbox needs to read. Each is
# bind-mounted read-only at its real path so compilers, runtimes, and VCS
# keep working without exposing secrets. Mirrors the enumeration the
# seatbelt provider re-allows under /Users, adapted to Linux paths (no
# `Library/Caches`). Each entry is resolved as: an absolute path as-is;
# a relative path against $HOME; a `~`-prefixed path via expanduser (leading
# `~` only, so a leading `~` works for entries the rest of the list keeps as
# relative). Nonexistent ones are skipped at argv-build time. Extend via
# `read_paths` in config.toml for project- or user-specific paths.
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
    # Node
    ".npmrc",
    ".local/lib/node_modules",
    ".cache/corepack",
    # Rust
    ".cargo",
    ".rustup",
    # Go
    ".cache/go-build",
    "go",
    "go.work",
    ".cache/gopls",
    # Nix
    ".nix-profile/bin",
    "/etc/nix",
    ".config/nix",
)


# Toolchain cache subpaths that a coding sandbox must WRITE to.
# Compilers and language servers (gopls, rust-analyzer) build metadata into
# these caches; a read-only bind breaks type checking and go-to-definition
# even though the source is readable. Each entry is resolved as: an absolute
# path as-is; a relative path against $HOME; a `~`-prefixed path via
# expanduser. Nonexistent ones are skipped at argv-build time. Symmetric to
# `_TOOL_READ_PATHS`.
_TOOL_WRITE_PATHS = (
    # Go: gopls writes compiled package metadata to the build cache. Without
    # write access it can't load views, so textDocument/definition returns null.
    ".cache/go-build",
    # gopls's own cache (typerefs, export data, diagnostics). Without write
    # access gopls logs errors on every request and degrades.
    ".cache/gopls",
    # uv: writes downloaded wheels and the sdist git index (sdists-v9/.git)
    # here; a read-only cache makes installs fail with "Operation not
    # permitted" on the .git index.
    ".cache/uv",
    # Nix
    "/nix/store",
    "/nix/var",
    ".cache/nix",
    ".local/share",
    ".local/state",
    f"/run/user/{os.getuid()}",
)


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
    import asyncio

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
    CLI/pytest callers). The resolved path is the one bind-mounted into the
    sandbox and used as `--chdir`, so it must match the kernel's view.
    """
    ctx = get_server_project_context()
    if ctx is not None:
        base = ctx.project_root if ctx.project_root is not None else ctx.user_cwd
        return base.resolve()
    return Path.cwd().resolve()


def _resolve_path(rel: str) -> Path:
    """Resolve one `_TOOL_*_PATHS` entry to a canonical candidate path.

    An absolute path (``/nix/store``) is used as-is; a relative path
    (``.cargo``) is joined under ``$HOME``; a ``~``-prefixed path is expanded
    via ``Path.expanduser`` (which honors only a *leading* ``~``, so ``~`` is
    only meaningful as the first segment). The result is NOT yet checked for
    existence or resolved to its real-path form -- callers do that after the
    existence filter, so a missing tool generates no bind and no noise.
    """
    # Expand a leading `~` first: pathlib's `/` operator discards the left
    # operand when the right is absolute, so `home / "~/.x"` would join to a
    # literal `~` segment -- expand before any join instead.
    p = Path(rel).expanduser()
    if p.is_absolute():
        return p
    return Path.home() / p


def _tool_read_paths() -> list[str]:
    """Resolve `_TOOL_READ_PATHS`, returning existing paths.

    Absolute entries are used as-is; relative entries are resolved against
    ``$HOME``; ``~``-prefixed entries are expanded (leading ``~`` only).
    Nonexistent paths are silently skipped (a tool not installed shouldn't
    generate noise or a failed bind). Unlike the seatbelt provider there is
    no `/Users` filter -- on Linux the home root is `/home/<user>` (or
    wherever `$HOME` points), and all of these are meant to be bound.

    Symlinks are NOT followed: the path is bound as-written (expanded, but
    not resolved) so the bind destination matches the symlink path that the
    inherited host ``PATH`` and ``$HOME``-relative lookups reference. ``bwrap``
    follows a symlink *source* itself, so binding ``~/.nix-profile/bin`` (a
    symlink into ``/nix/var/...``) still mounts the real nix binaries at the
    symlink path -- the agent's PATH entry ``~/.nix-profile/bin`` then
    resolves inside the sandbox. ``.resolve()`` would instead bind at
    ``/nix/var/nix/profiles/default/bin``, which PATH never references, so
    the agent would get ``command not found`` even though the bind exists
    (and that resolved path is also shadowed by the later writable
    ``/nix/var`` bind).
    """
    out: list[str] = []
    for rel in _TOOL_READ_PATHS:
        resolved = _resolve_path(rel)
        if not resolved.exists():
            continue
        out.append(str(resolved))
    return out


def _tool_write_paths() -> list[str]:
    """Resolve `_TOOL_WRITE_PATHS`, returning existing paths.

    Same resolution and skip rule as `_tool_read_paths`: absolute paths
    as-is, relative paths against ``$HOME``, ``~``-prefixed via expanduser;
    nonexistent paths are dropped (a tool not installed generates no cache
    to write). Symlinks are NOT followed, for the same reason as
    `_tool_read_paths`: the bind must land at the path the toolchain looks
    up, not the symlink's resolved target.
    """
    out: list[str] = []
    for rel in _TOOL_WRITE_PATHS:
        resolved = _resolve_path(rel)
        if not resolved.exists():
            continue
        out.append(str(resolved))
    return out


def _existing(paths: list[str]) -> list[str]:
    """Return the canonical, existing, de-duplicated subset of `paths`.

    `bwrap --ro-bind` fails on a missing source, so `read_paths` that don't
    exist are dropped here rather than forcing a hard error (a user might
    list a path that exists on one machine but not another). Resolves
    symlinks so the bind source matches the kernel's view.
    """
    seen: set[str] = set()
    out: list[str] = []
    for raw in paths:
        try:
            resolved = str(Path(raw).expanduser().resolve())
        except OSError:
            continue
        if resolved in seen or not Path(resolved).exists():
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def _bind_ancestor_args(
    path: str,
    already_bound: set[str],
) -> list[str]:
    """Pre-create read-only ancestor dirs for a bind target, in order.

    bwrap auto-creates any missing parent directories for a bind destination,
    but it makes them mode 0755 (writable). When a bind target lives several
    levels deep under the empty tmpfs root (e.g. ``/home/me/code/proj``),
    bwrap would otherwise create ``/home``, ``/home/me``, ``/home/me/code``
    as writable empty dirs -- so a sandboxed process could write a file at
    ``/home/me/evil.txt`` (inside the tmpfs, not on the host, but still
    inside the sandbox's writable surface, which a hard fence must close).
    Pre-creating each ancestor with ``--perms 0555 --dir <ancestor>`` makes
    those auto-created parents read-only, so the launch dir is the only
    writable bind it sets up. (Ancestors of the system ro-binds, e.g.
    ``/run`` for the ``/run/systemd/resolve`` bind, keep bwrap's default and
    may be writable, but as empty tmpfs scratch -- not host mounts -- they
    expose no host data.)

    Emits a ``--perms 0555 --dir <ancestor>`` triple for each ancestor of
    ``path`` from the root down to (but not including) ``path`` itself, for
    ancestors not already covered by an earlier bind (a bound path is its
    own subtree, so its ancestors are created by *its* chain). ``already_bound``
    is updated in place with each ancestor emitted so later binds skip it.

    Ancestors that are bound directly elsewhere in the argv (e.g. ``/usr`` is
    bound, so ``/usr/local/foo`` needs no ancestor pre-creation) are skipped
    by checking membership in ``already_bound``; the caller seeds it with the
    system ro-bind destinations and the launch dir.
    """
    parts = [p for p in path.split("/") if p]
    if not parts:
        return []
    argv: list[str] = []
    acc = ""
    # All ancestors except the final segment (the bind target itself).
    for part in parts[:-1]:
        acc += "/" + part
        if acc in already_bound:
            continue
        argv += ["--perms", "0555", "--dir", acc]
        already_bound.add(acc)
    return argv


class BubblewrapSandbox(BaseSandbox):
    """Local sandbox backend that runs every operation under `bwrap`.

    The launch directory is the read/write area, bind-mounted read-write at
    its real path. System trees (`/usr`, `/etc`, `/bin`, `/lib`, `/proc`,
    `/dev`, ...) are bind-mounted read-only so a process can bootstrap.
    Everything else -- including all of `$HOME` except the launch dir and a
    curated list of toolchain/config subpaths -- is not mounted, so it is
    invisible: `~/.ssh`, `~/.aws`, other users' files, `/var`, `/run`
    sockets, etc. are absent. `read_paths` adds extra read-only binds.
    Network is allowed by default (host net shared) unless `network=False`
    was passed at creation. Each command runs with a minimal, secret-free
    environment (PATH, HOME, TMPDIR, SHELL, LANG) instead of inheriting
    dcode's process environment.

    `upload_files` and `download_files` are routed through `execute` (via
    base64 over the sandboxed process's stdin/stdout) so the mount namespace
    fences them exactly like ordinary commands -- there are no direct host
    filesystem calls in the provider process, and a path outside the launch
    dir simply isn't mounted, so the transport fails the same way a manual
    `cat` would.

    The `BaseSandbox.grep`/`agrep` inherited implementation is used
    unmodified: it builds `grep -rHnFZ`, and on Linux GNU grep treats `-Z`
    as `--null` (a NUL after the filename), which the base `_parse_grep_output`
    expects. No override is needed (unlike the macOS seatbelt provider, where
    BSD grep reads `-Z` as `--decompress`).
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
        # Resolve to the canonical path now: it is the source of the rw bind
        # mount and the `--chdir` target, so it must match the kernel's view
        # of the path. get_or_create already resolves, but direct
        # construction (tests, validation suite) may not.
        self._launch = launch_dir.resolve()
        self._network = network
        self._read_paths = list(read_paths) if read_paths else []
        # The bwrap argv is a pure function of construction-time state, so it
        # is recomputed on every call (there is no on-disk profile to cache,
        # unlike the seatbelt provider's SBPL file). Concurrent executes are
        # independent bwrap invocations sharing no state, so no lock is
        # needed.

    @property
    def id(self) -> str:
        return self._id

    def _env(self) -> dict[str, str]:
        # Don't inherit dcode's environment: it typically carries API keys and
        # tokens (OPENAI_API_KEY, GITHUB_TOKEN, ...) the sandboxed code must not
        # see or exfiltrate. Pass a minimal, secret-free environment instead.
        # HOME is the real user home, not the launch dir: toolchains resolve
        # their caches relative to HOME (Go: ~/go, ~/.cache/go-build; Rust:
        # ~/.cargo; ...), and the mount namespace -- not HOME -- is the fence
        # that keeps secrets (~/.ssh, ~/.aws, ...) invisible (unmounted).
        #
        # PATH is *inherited* from the host rather than hardcoded. PATH is not
        # a secret -- it is just a list of directories -- and the curated
        # read-only binds (`_tool_read_paths`) mount the user's toolchains
        # (~/.nix-profile/bin, ~/.cargo/bin, ~/.local/bin, ...) at their real
        # paths, but those binaries are unreachable unless their dirs are on
        # PATH. A hardcoded PATH would make the agent blind to the very tools
        # the binds were added to expose; inheriting the host PATH lets the
        # agent use the same tools the user does. (PATH entries whose dirs are
        # not mounted in the sandbox are simply dead entries -- a `command not
        # found`, not a leak or a hole.)
        return {
            "PATH": os.environ.get(
                "PATH", "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/snap/bin"
            ),
            "HOME": str(Path.home()),
            # TMPDIR inside the launch dir (a writable area); created lazily by
            # the sandboxed command itself, not by the provider process.
            "TMPDIR": str(self._launch / ".tmp"),
            "SHELL": "/bin/sh",
            "LANG": "en_US.UTF-8",
        }

    def _bwrap_argv(self, command: str) -> list[str]:
        """Build the full `bwrap ... sh -c <command>` argv for `command`.

        The argv is a pure function of construction-time state (launch dir,
        network flag, read_paths), so it is safe to rebuild on every call.
        Order: namespaces -> lockdown -> env (via subprocess env=, not bwrap
        flags) -> system ro-binds -> proc/dev -> launch dir rw bind ->
        tool/read ro-binds -> chdir -> command.
        """
        argv: list[str] = ["bwrap"]

        # --- Namespaces -----------------------------------------------------
        # --unshare-user-try: create a user namespace if possible (required
        #   for an unprivileged user to create the other namespaces; skipped
        #   only when running as root with caps, in which case the others
        #   still work). v0.12 dropped setuid mode, so unprivileged userns is
        #   mandatory for non-root -- if the kernel disallows it, bwrap fails
        #   and the error surfaces on the first execute().
        # --unshare-ipc / --unshare-pid / --unshare-uts / --unshare-cgroup-try:
        #   isolate SysV IPC, PIDs (sandbox can't see host processes), the
        #   hostname, and cgroups (best-effort). Network is shared by default.
        argv += [
            "--unshare-user-try",
            "--unshare-ipc",
            "--unshare-pid",
            "--unshare-uts",
            "--unshare-cgroup-try",
        ]
        if not self._network:
            argv += ["--unshare-net"]

        # --- Lockdown -------------------------------------------------------
        # --new-session: setsid(), detaching from the controlling terminal so
        #   the sandbox can't inject keystrokes (TIOCSTI, CVE-2017-5226).
        # --die-with-parent: kill the sandbox if the provider process dies,
        #   so a crashed dcode doesn't leave a runaway sandbox.
        argv += ["--new-session", "--die-with-parent"]

        # --- Environment (secret-scrubbed, injected via bwrap flags) ---------
        # --clearenv wipes every inherited variable (including dcode's API
        # keys/tokens) before the sandboxed process runs; --setenv then installs
        # the minimal secret-free env. This is injected via bwrap flags rather
        # than subprocess `env=` so the bwrap *launcher* itself runs under the
        # inherited host environment -- that lets the host find the `bwrap`
        # binary wherever it lives (nix profile, /usr/bin, ...), while the
        # sandboxed process still gets only the scrubbed env. (The seatbelt
        # provider can pass the minimal env via `env=` because sandbox-exec is
        # always at /usr/bin; bwrap has no such fixed location.)
        argv += ["--clearenv"]
        for key, value in self._env().items():
            argv += ["--setenv", key, value]

        # --- System read-only binds ----------------------------------------
        # The sandbox root is an empty tmpfs; these make a process able to
        # bootstrap (dynamic linker, libc, /etc, /proc, /dev) and the common
        # toolchains reachable. All read-only: the sandbox cannot modify the
        # host system image. A bound subtree covers its descendants, so seed
        # the ancestor-tracking set with every system bind destination --
        # a deep path under /usr then needs no ancestor pre-creation.
        bound: set[str] = set()
        for src, dst, try_ in _SYSTEM_RO_BINDS:
            argv += ["--ro-bind-try" if try_ else "--ro-bind", src, dst]
            bound.add(dst)
        argv += ["--proc", "/proc", "--dev", "/dev"]
        bound.update({"/proc", "/dev"})
        # A writable /tmp: Go's toolchain (`go list`, `go build`) creates temp
        # work dirs under /tmp (via /tmp, not TMPDIR). Without a writable /tmp
        # package loading fails with "mkdir /tmp/go-build*: operation not
        # permitted".
        argv += ["--tmpfs", "/tmp"]
        bound.add("/tmp")

        # --- Launch dir (read/write) ---------------------------------------
        # Bind the launch dir at its real path, read-write, so the agent
        # reads/writes the actual project with zero config and operates on
        # the real paths it is told (no virtual root translation). Its
        # ancestors under the tmpfs root are pre-created read-only (0555) so
        # the auto-created parents can't be written into (see
        # `_bind_ancestor_args`): only the launch dir itself is writable.
        launch_str = str(self._launch)
        argv += _bind_ancestor_args(launch_str, bound)
        argv += ["--bind", launch_str, launch_str]
        bound.add(launch_str)

        # --- Curated tool paths under $HOME (read-only) --------------------
        # Compilers/runtimes/VCS caches keep working without exposing the
        # rest of $HOME. Bound at their real paths (HOME inside the sandbox
        # is the launch dir, so $HOME-relative lookups won't find these --
        # same behavior as the seatbelt provider; tools configured with
        # absolute paths or the relevant *_HOME env vars reach them).
        # Ancestors are pre-created read-only too, so a bound `~/.cargo`
        # doesn't open a writable `~/.cargo/registry/..` escape path; if an
        # ancestor is already bound (e.g. the launch dir sits under $HOME),
        # it is skipped via the shared `bound` set.
        for p in _tool_read_paths():
            argv += _bind_ancestor_args(p, bound)
            argv += ["--ro-bind-try", p, p]
            bound.add(p)

        # --- Curated tool write paths under $HOME (read/write) -------------
        # Compilers and language servers (gopls, rust-analyzer) build metadata
        # into these caches; a read-only bind breaks type checking and
        # go-to-definition even though the source is readable. Bound
        # read-write at their real paths. Ancestors are pre-created read-only
        # (0555) so only the cache dir itself is writable, mirroring the
        # read-only tool path handling above.
        for p in _tool_write_paths():
            argv += _bind_ancestor_args(p, bound)
            argv += ["--bind", p, p]
            bound.add(p)

        # --- User read_paths (read-only) -----------------------------------
        for p in _existing(self._read_paths):
            argv += _bind_ancestor_args(p, bound)
            argv += ["--ro-bind", p, p]
            bound.add(p)

        # --- Working directory + command ----------------------------------
        argv += ["--chdir", launch_str, "sh", "-c", command]
        return argv

    def _run(
        self,
        argv: list[str],
        *,
        stdin: bytes | None = None,
        timeout: int | None = None,
    ) -> tuple[bytes, bytes, int | None]:
        """Run `argv` under bwrap, returning raw stdout/stderr/exit_code.

        Bytes throughout (no `text=True`) so binary file content round-trips
        exactly through the base64 transport used by upload/download. A
        timeout yields the partial output captured so far and ``None`` exit.
        `--die-with-parent` ensures a timed-out/kill bwrap tears down its
        sandboxed children.

        bwrap is launched with the **inherited** host environment (no
        `env=` override) so the host can find the `bwrap` binary wherever it
        lives (nix, /usr/bin, ...). The secret-scrubbed sandbox env is injected
        via bwrap's own `--clearenv`/`--setenv` flags inside the argv, so the
        sandboxed process still gets only the minimal, secret-free env.
        """
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                timeout=timeout or _DEFAULT_TIMEOUT,
                check=False,
                input=stdin,
            )
        except subprocess.TimeoutExpired as exc:
            return bytes(exc.stdout or b""), bytes(exc.stderr or b""), None
        return proc.stdout, proc.stderr, proc.returncode

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        out, err, rc = self._run(self._bwrap_argv(command), timeout=timeout)
        output = (out + err).decode("utf-8", errors="replace")
        if rc is None:
            output += f"\n[command timed out after {timeout or _DEFAULT_TIMEOUT}s]"
        return ExecuteResponse(output=output, exit_code=rc)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        responses: list[FileUploadResponse] = []
        for path, content in files:
            qpath = shlex.quote(path)
            qparent = shlex.quote(str(Path(path).parent))
            # mkdir the parent (no-op if it exists) then decode base64 from
            # stdin into the target. The write fence (target must be inside
            # the launch dir, the only writable mount) is enforced by the
            # mount namespace, not here. GNU base64 -d reads stdin and
            # decodes; `mkdir -p` needs the parent to be writable (it is,
            # inside the launch dir).
            command = f"mkdir -p {qparent} && base64 -d > {qpath}"
            _out, err, rc = self._run(
                self._bwrap_argv(command),
                stdin=base64.b64encode(content),
            )
            if rc == 0:
                responses.append(FileUploadResponse(path=path, error=None))
            else:
                detail = err.decode("utf-8", errors="replace").strip()
                responses.append(
                    FileUploadResponse(path=path, error=detail or f"exit {rc}")
                )
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        responses: list[FileDownloadResponse] = []
        for path in paths:
            qpath = shlex.quote(path)
            # GNU base64 encodes the given file to stdout; -w0 disables line
            # wrapping for a compact stream (the decoder handles newlines
            # either way). A path outside the launch dir is simply not
            # mounted, so this fails with "No such file" -- the fence, not a
            # provider-side check.
            out, err, rc = self._run(self._bwrap_argv(f"base64 -w0 {qpath}"))
            if rc == 0:
                try:
                    content = base64.b64decode(out)
                except (ValueError, binascii.Error):
                    detail = "base64 decode failed"
                    responses.append(
                        FileDownloadResponse(path=path, content=None, error=detail)
                    )
                    continue
                responses.append(
                    FileDownloadResponse(path=path, content=content, error=None)
                )
                continue
            detail = err.decode("utf-8", errors="replace").lower()
            if "no such file" in detail or "not found" in detail:
                responses.append(
                    FileDownloadResponse(
                        path=path, content=None, error="file_not_found"
                    )
                )
            else:
                msg = err.decode("utf-8", errors="replace").strip() or f"exit {rc}"
                responses.append(
                    FileDownloadResponse(path=path, content=None, error=msg)
                )
        return responses


class BubblewrapProvider(SandboxProvider):
    """SandboxProvider that creates/deletes local bwrap-isolated sandboxes."""

    @property
    def metadata(self) -> SandboxProviderMetadata:
        # `working_dir` is the path the model is told to operate in. It is read
        # by the registry during discovery (which instantiates this provider on
        # the langgraph-guarded event loop), so it must not make blocking fs
        # calls -- `_launch_dir_raw` only reads env vars. The canonical path
        # is resolved in `get_or_create` (offloaded to a worker thread).
        return SandboxProviderMetadata(
            name="bubblewrap",
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
    ) -> BubblewrapSandbox:
        def _create() -> BubblewrapSandbox:
            # Validate the id before probing for bwrap: an invalid id is a
            # caller bug regardless of whether bwrap is installed, and the
            # validation tests must pass on machines without bwrap (the
            # validation suite runs on any OS).
            if sandbox_id is not None and not _ID_PATTERN.match(sandbox_id):
                msg = (
                    f"Invalid sandbox_id {sandbox_id!r}: must match "
                    "[A-Za-z0-9._-]+ (no path separators or whitespace)."
                )
                raise ValueError(msg)
            if shutil.which("bwrap") is None:
                msg = (
                    "bwrap is not available. Install bubblewrap (e.g. "
                    "`apt install bubblewrap`, `dnf install bubblewrap`, or "
                    "`pacman -S bubblewrap`) and ensure unprivileged user "
                    "namespaces are enabled. This provider only works on Linux."
                )
                raise RuntimeError(msg)
            sid = sandbox_id or f"bubblewrap-{uuid.uuid4().hex[:12]}"
            launch = _resolve_launch_dir()
            return BubblewrapSandbox(
                sid, launch, network=network, read_paths=read_paths
            )

        # Offload the blocking fs calls (`shutil.which`/`Path.resolve`) when
        # called on the langgraph-guarded event loop at graph build time; run
        # inline for sync callers (pytest, CLI).
        return _offload_if_running_loop(_create)

    def delete(self, *, sandbox_id: str, **kwargs: Any) -> None:
        # No per-sandbox workspace and no on-disk profile exist -- there is
        # nothing to clean up. The id is still validated for API parity with
        # the seatbelt provider (a friendly fast-fail, not a security boundary).
        if not _ID_PATTERN.match(sandbox_id):
            msg = (
                f"Invalid sandbox_id {sandbox_id!r}: must match [A-Za-z0-9._-]+ "
                "(no path separators or whitespace)."
            )
            raise ValueError(msg)
