# dcode-seatbelt-sandbox

Local, zero-cloud sandbox provider for `deepagents-code` (`dcode`) on macOS,
backed by `sandbox-exec` (Seatbelt). Mirrors `dcode-bubblewrap-sandbox`, but
for macOS instead of Linux.

## Requirements

- macOS (`sandbox-exec` ships with the OS)
- Python >= 3.10

## Install with uv

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dcode]"
```

## Use

```bash
dcode --sandbox seatbelt
```

Reuse an existing sandbox workspace across runs with a custom id (allowed
characters: `A-Z`, `a-z`, `0-9`, `.`, `_`, `-`):

```bash
dcode --sandbox seatbelt --sandbox-id my-project
```

## What's isolated

- Filesystem: **writes** restricted to a per-sandbox workspace directory (and
  `/dev/null` for shell redirections). **Reads** are allowed broadly across
  the host (system paths, Homebrew on Apple Silicon at `/opt/homebrew`, the
  shared dyld cache, frameworks, locale data) -- this is required for a
  process to bootstrap under `(deny default)`: the dynamic linker needs read
  access to its supporting files before `main()` runs. **But reads of
  `/Users` (every user's home directory) are denied** -- `~/.ssh`, `~/Desktop`,
  `~/.aws`, other users' files, etc. are off-limits. The directory you launched
  `dcode` from is re-allowed automatically, so the agent can read its own
  project with zero config. `read_paths` (via
  `[sandboxes.providers.seatbelt.params]` in `~/.deepagents/config.toml`) adds
  *extra* readable paths on top of that, if you need the agent to reach
  sibling projects or other directories:

  ```toml
  [sandboxes.providers.seatbelt.params]
  read_paths = ["/Users/you/code/other-project"]
  ```

  System files outside `/Users` stay readable (they are Apple's, not yours,
  and reinstallable). The security boundary is write confinement plus the
  `/Users` read fence, not read isolation of the whole host.
- Environment: commands run with a minimal, secret-free environment (PATH,
  HOME, TMPDIR, SHELL, LANG). dcode's own environment (typically containing
  API keys and tokens like `OPENAI_API_KEY`, `GITHUB_TOKEN`) is **not**
  inherited, so sandboxed code can't `printenv` them or exfiltrate them.
  `HOME` points at the real user home (not the launch dir): toolchains
  resolve their caches relative to `HOME` (Go: `~/go`, Rust: `~/.cargo`),
  and the SBPL profile -- not `HOME` -- is the fence that keeps the home's
  secrets (`~/.ssh`, `~/.aws`, ...) unreadable. `PATH` is the one host
  variable that *is* inherited (with a sane fallback) -- it's just a list of
  directories, not a secret, and inheriting it lets the agent reach the same
  toolchain binaries (`cargo`, `go`, `nix`, ...) that the curated read
  re-allows under `/Users` make executable.
- Network: denied by default. Pass `network=True` via
  `[sandboxes.providers.seatbelt.params]` in `~/.deepagents/config.toml` to
  allow it (all-or-nothing -- no per-host allowlisting).
- Process: the Seatbelt profile is stored outside the workspace so the
  sandboxed process can't read its own confinement rules; `process-exec` is
  allowed globally (any system binary and any binary written into the
  workspace) so compilers and build tools keep working.

No PID/mount namespaces (unlike bubblewrap) -- this is a syscall/resource
filter, not container-style isolation. `sandbox-exec` is deprecated by Apple
with no official replacement, but still functional and patched on current
macOS. This is protection against accidental scope creep, not a hard boundary
against adversarial code.
