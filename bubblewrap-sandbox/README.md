# dcode-bubblewrap-sandbox

Local, zero-cloud sandbox provider for `deepagents-code` (`dcode`) on Linux,
backed by [bubblewrap](https://github.com/containers/bubblewrap) (`bwrap`).
Mirrors `dcode-seatbelt-sandbox`, but for Linux instead of macOS.

## Requirements

- Linux with `bwrap` installed (e.g. `apt install bubblewrap`,
  `dnf install bubblewrap`, or `pacman -S bubblewrap`)
- Unprivileged user namespaces enabled (required by bubblewrap v0.12+, which
  dropped the legacy setuid mode)
- Python >= 3.13

## Install with uv

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dcode]"
```

## Use

```bash
dcode --sandbox bubblewrap
```

Reuse an existing sandbox workspace across runs with a custom id (allowed
characters: `A-Z`, `a-z`, `0-9`, `.`, `_`, `-`):

```bash
dcode --sandbox bubblewrap --sandbox-id my-project
```

## What's isolated

- **Filesystem**: the sandbox root is an empty tmpfs. The directory you
  launched `dcode` from (the *launch dir*) is bind-mounted **read-write** at
  its real path, so the agent reads and writes the project with zero config
  and operates on the real paths it is told (no virtual root translation).
  System trees needed to bootstrap a process (`/usr`, `/etc`, `/bin`, `/lib`,
  `/lib64`, `/sbin`, `/opt`, `/proc`, `/dev`) are bind-mounted **read-only**.
  A curated list of toolchain/config subpaths under `$HOME` (`.gitconfig`,
  `.cargo`, `.rustup`, `.local/share/uv`, `.npmrc`, `go`, ...) is also bound
  **read-only** so compilers, runtimes, and VCS keep working without exposing
  secrets. **Everything else -- including all of `$HOME` except the launch
  dir and that curated list -- is simply not mounted**, so it is invisible
  rather than merely denied: `~/.ssh`, `~/.aws`, other users' files,
  `/var`, `/run` sockets, etc. are absent. `read_paths` (via
  `[sandboxes.providers.bubblewrap.params]` in `~/.deepagents/config.toml`)
  adds *extra* read-only bind mounts on top of that, if you need the agent
  to reach sibling projects or other directories:

  ```toml
  [sandboxes.providers.bubblewrap.params]
  read_paths = ["/home/you/code/other-project"]
  ```

  Writes are confined to the launch dir: nothing else is mounted writable,
  so every other path is either read-only (the system binds) or
  non-existent (everything else).
- **Environment**: commands run with a minimal, secret-free environment
  (PATH, HOME, TMPDIR, SHELL, LANG). dcode's own environment (typically
  containing API keys and tokens like `OPENAI_API_KEY`, `GITHUB_TOKEN`) is
  **not inherited**, so sandboxed code can't `printenv` them or exfiltrate
  them. `HOME` points at the launch dir (the writable area): the sandbox's
  "home" is the project, and the real home's secrets are not mounted. `PATH`
  is the one host variable that *is* inherited (with a sane fallback) -- it's
  just a list of directories, not a secret, and inheriting it lets the agent
  reach the same toolchain binaries (`nix`, `cargo`, `go`, ...) that the
  read-only binds below mount into the sandbox.
- **Network**: **allowed by default** (the host network namespace is shared
  -- coding work needs it: pip, npm, git, tests, dev servers). Pass
  `network=False` via `[sandboxes.providers.bubblewrap.params]` in
  `~/.deepagents/config.toml` to opt out, which adds `--unshare-net`
  (loopback-only).
- **Process**: bubblewrap creates PID, IPC, UTS, and (best-effort) cgroup
  namespaces, plus a private mount namespace with an empty root. The
  sandbox cannot see or signal host processes. `--new-session` (setsid)
  detaches from the controlling terminal, blocking the TIOCSTI
  terminal-injection attack (CVE-2017-5226). `--die-with-parent` kills the
  sandbox if the provider process dies.

Unlike seatbelt (a syscall/resource filter with no namespaces), bubblewrap
gives real container-style isolation (PID/mount/IPC/UTS namespaces, a private
mount root). This is a stronger boundary, but still should not be treated as
a hard boundary against adversarial code with kernel exploits.

## Notes on tool paths

`HOME` inside the sandbox is set to the launch dir (the writable area), not
your real home. The curated tool paths are bound read-only at their **real
absolute paths** (e.g. `/home/you/.cargo`), not at `$HOME/.cargo`. Tools that
locate their config via an absolute path or a dedicated `*_HOME` env var
(`CARGO_HOME`, `GOPATH`, `npm_config_prefix`, ...) will find them. Tools
that look them up purely via `$HOME`-relative paths will not, by design --
binding the real `$HOME` would re-expose `~/.ssh` and `~/.aws`.
