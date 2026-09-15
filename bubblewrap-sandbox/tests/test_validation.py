"""Bubblewrap provider tests: sandbox_id validation, the `_existing` /
`_tool_read_paths` helpers, and the BaseSandbox abstract surface. Some cases
exercise `bwrap`, so these require Linux.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from dcode_bubblewrap_sandbox import BubblewrapSandbox
from dcode_bubblewrap_sandbox.provider import (
    _ID_PATTERN,
    _SYSTEM_RO_BINDS,
    _existing,
    _tool_read_paths,
)

pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="bubblewrap tests require Linux",
)

# sandbox_id validation


class TestIdValidation:
    """`sandbox_id` never reaches disk for bubblewrap (no profile/workspace),
    so the regex is API parity + friendly error, not a security boundary."""

    @pytest.mark.parametrize(
        "bad",
        [
            "a/b",  # path separator
            "a\\b",  # backslash (hazard on Windows-shared mounts)
            "",  # empty
            "foo bar",  # whitespace
            "foo\tbar",  # tab
            "foo\nbar",  # newline
            "foo;bar",  # shell metachar
            "foo|bar",
            "foo&bar",
            "foo$bar",
            "foo`bar`",
            "foo(bar)",
            "foo{bar}",
            "foo>bar",
            "foo*bar",
            "foo?bar",
            "foo[bar]",
            "\x00",  # null byte
        ],
    )
    def test_rejects_invalid_ids(self, bad: str) -> None:
        assert _ID_PATTERN.match(bad) is None

    @pytest.mark.parametrize(
        "ok",
        [
            "bubblewrap-abc123",
            "my_project",
            "my.project",
            "a",
            "A1-B2.c3",
            "-n",  # leading dash: safe, id is never a CLI arg
            ".hidden",
            "..",
        ],
    )
    def test_accepts_valid_ids(self, ok: str) -> None:
        assert _ID_PATTERN.match(ok) is not None


# _existing (read_paths normalizer)


class TestExisting:
    """`_existing` drops nonexistent paths, resolves symlinks, and dedupes:
    `bwrap --ro-bind` fails on a missing source."""

    def test_drops_nonexistent(self, tmp_path: Path) -> None:
        real = tmp_path / "real"
        real.mkdir()
        missing = tmp_path / "missing"
        out = _existing([str(real), str(missing)])
        assert str(real.resolve()) in out
        assert str(missing) not in out

    def test_dedupes(self, tmp_path: Path) -> None:
        d = tmp_path / "d"
        d.mkdir()
        out = _existing([str(d), str(d), str(d.resolve())])
        assert out.count(str(d.resolve())) == 1

    def test_resolves_symlinks(self, tmp_path: Path) -> None:
        target = tmp_path / "target"
        target.mkdir()
        link = tmp_path / "link"
        link.symlink_to(target)
        out = _existing([str(link)])
        assert out == [str(target.resolve())]

    def test_expands_user(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        d = tmp_path / "proj"
        d.mkdir()
        out = _existing(["~/proj"])
        assert out == [str(d.resolve())]


# _tool_read_paths


class TestToolReadPaths:
    """`_tool_read_paths` resolves `_TOOL_READ_PATHS` against $HOME and skips
    nonexistent entries."""

    def test_skips_nonexistent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        # Nothing created -> no tool paths exist -> empty.
        assert _tool_read_paths() == []

    def test_returns_existing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / ".cargo").mkdir()
        (tmp_path / ".config" / "git").mkdir(parents=True)
        out = _tool_read_paths()
        assert str((tmp_path / ".cargo").resolve()) in out
        assert str((tmp_path / ".config" / "git").resolve()) in out

    def test_includes_files(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # A file (e.g. ~/.gitconfig) is included: bwrap --ro-bind works on files.
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / ".gitconfig").write_text("[user]\n")
        out = _tool_read_paths()
        assert str((tmp_path / ".gitconfig").resolve()) in out


# system ro-bind table


class TestSystemRoBinds:
    """Mounts essentials read-only and exposes DNS without all of /run."""

    def test_usr_and_etc_are_hard_binds(self) -> None:
        binds = {src: try_ for src, _dst, try_ in _SYSTEM_RO_BINDS}
        assert binds["/usr"] is False
        assert binds["/etc"] is False

    def test_merged_usr_dirs_are_try_binds(self) -> None:
        binds = {src: try_ for src, _dst, try_ in _SYSTEM_RO_BINDS}
        for sym in ("/bin", "/lib", "/lib64", "/sbin"):
            if sym in binds:
                assert binds[sym] is True

    def test_run_exposed_only_via_resolv_subdirs(self) -> None:
        run_binds = [
            dst for _src, dst, _try in _SYSTEM_RO_BINDS if dst.startswith("/run")
        ]
        # Only the two resolv.conf-related subdirs, not /run itself.
        assert "/run/systemd/resolve" in run_binds
        assert "/run/resolvconf" in run_binds
        assert "/run" not in run_binds


# BaseSandbox abstract surface


class TestBaseSandboxSurface:
    """Abstract members are implemented; derived methods come from BaseSandbox."""

    def test_sandbox_instantiates_and_has_abstract_members(
        self, tmp_path: Path
    ) -> None:
        sb = BubblewrapSandbox("surface-test", tmp_path)
        assert sb.id == "surface-test"
        for name in ("execute", "upload_files", "download_files"):
            assert callable(getattr(sb, name)), name

    def test_derived_methods_present(self, tmp_path: Path) -> None:
        sb = BubblewrapSandbox("surface-test", tmp_path)
        for name in ("ls", "read", "write", "edit", "delete", "grep", "glob"):
            assert callable(getattr(sb, name)), name


# argv construction (no bwrap needed)


class TestArgvConstruction:
    """`_bwrap_argv` is a pure function of construction state; assert its shape
    without running bwrap. Mirrors the seatbelt profile-text tests."""

    def test_argv_has_namespaces_and_lockdown(self, tmp_path: Path) -> None:
        sb = BubblewrapSandbox("argv-test", tmp_path)
        argv = sb._bwrap_argv("echo hi")
        assert argv[0] == "bwrap"
        assert "--unshare-user-try" in argv
        assert "--unshare-ipc" in argv
        assert "--unshare-pid" in argv
        assert "--unshare-uts" in argv
        assert "--unshare-cgroup-try" in argv
        assert "--new-session" in argv
        assert "--die-with-parent" in argv

    def test_argv_shares_net_by_default(self, tmp_path: Path) -> None:
        sb = BubblewrapSandbox("argv-test", tmp_path, network=True)
        argv = sb._bwrap_argv("echo hi")
        assert "--unshare-net" not in argv

    def test_argv_unshares_net_when_disabled(self, tmp_path: Path) -> None:
        sb = BubblewrapSandbox("argv-test", tmp_path, network=False)
        argv = sb._bwrap_argv("echo hi")
        assert "--unshare-net" in argv

    def test_argv_binds_launch_dir_rw(self, tmp_path: Path) -> None:
        sb = BubblewrapSandbox("argv-test", tmp_path)
        argv = sb._bwrap_argv("echo hi")
        resolved = str(tmp_path.resolve())
        # --bind is rw; the launch dir appears as both src and dst at its real path.
        idx = argv.index("--bind")
        assert argv[idx + 1] == resolved
        assert argv[idx + 2] == resolved

    def test_argv_binds_system_ro(self, tmp_path: Path) -> None:
        sb = BubblewrapSandbox("argv-test", tmp_path)
        argv = sb._bwrap_argv("echo hi")
        # /usr and /etc are hard ro-binds (not --ro-bind-try).
        assert "--ro-bind" in argv
        usr_idx = argv.index("--ro-bind")
        assert argv[usr_idx + 1] == "/usr"
        assert argv[usr_idx + 2] == "/usr"
        # --proc and --dev present.
        assert "--proc" in argv
        assert "--dev" in argv

    def test_argv_chdirs_to_launch_dir(self, tmp_path: Path) -> None:
        sb = BubblewrapSandbox("argv-test", tmp_path)
        argv = sb._bwrap_argv("echo hi")
        resolved = str(tmp_path.resolve())
        chdir_idx = argv.index("--chdir")
        assert argv[chdir_idx + 1] == resolved

    def test_argv_ends_with_sh_c_command(self, tmp_path: Path) -> None:
        sb = BubblewrapSandbox("argv-test", tmp_path)
        argv = sb._bwrap_argv("echo hi")
        assert argv[-3:] == ["sh", "-c", "echo hi"]

    def test_argv_includes_read_paths_ro(self, tmp_path: Path) -> None:
        extra = tmp_path / "extra"
        extra.mkdir()
        sb = BubblewrapSandbox("argv-test", tmp_path, read_paths=[str(extra)])
        argv = sb._bwrap_argv("echo hi")
        assert "--ro-bind" in argv
        resolved = str(extra.resolve())
        # The read_path bind appears (distinct from the /usr /etc system binds).
        binds = [
            (argv[i + 1], argv[i + 2])
            for i in range(len(argv) - 2)
            if argv[i] == "--ro-bind"
        ]
        assert (resolved, resolved) in binds

    def test_argv_skips_nonexistent_read_paths(self, tmp_path: Path) -> None:
        sb = BubblewrapSandbox(
            "argv-test", tmp_path, read_paths=[str(tmp_path / "nope")]
        )
        argv = sb._bwrap_argv("echo hi")
        assert str(tmp_path / "nope") not in argv


# environment scrubbing (no bwrap needed)


class TestEnvScrubbing:
    """`_env` returns a minimal, secret-free environment: dcode's own env
    (with API keys/tokens) is not inherited."""

    def test_minimal_env_only(self, tmp_path: Path) -> None:
        sb = BubblewrapSandbox("env-test", tmp_path)
        env = sb._env()
        assert set(env) == {"PATH", "HOME", "TMPDIR", "SHELL", "LANG"}

    def test_home_is_real_user_home(self, tmp_path: Path) -> None:
        # HOME is the real user home, not the launch dir: toolchains resolve
        # their caches relative to HOME (~/go, ~/.cargo, ~/.cache/...), and
        # the mount namespace, not HOME, is the fence that keeps secrets
        # (~/.ssh, ~/.aws, ...) invisible (unmounted). See provider._env.
        sb = BubblewrapSandbox("env-test", tmp_path)
        assert sb._env()["HOME"] == str(Path.home())

    def test_tmpdir_inside_launch_dir(self, tmp_path: Path) -> None:
        sb = BubblewrapSandbox("env-test", tmp_path)
        assert sb._env()["TMPDIR"] == str(tmp_path.resolve() / ".tmp")


# provider id validation (no bwrap needed)


class TestProviderIdValidation:
    def test_rejects_separator(self) -> None:
        from dcode_bubblewrap_sandbox.provider import BubblewrapProvider

        with pytest.raises(ValueError):
            BubblewrapProvider().get_or_create(sandbox_id="a/b")

    def test_rejects_whitespace(self) -> None:
        from dcode_bubblewrap_sandbox.provider import BubblewrapProvider

        with pytest.raises(ValueError):
            BubblewrapProvider().get_or_create(sandbox_id="foo bar")

    def test_delete_rejects_bad_id(self) -> None:
        from dcode_bubblewrap_sandbox.provider import BubblewrapProvider

        with pytest.raises(ValueError):
            BubblewrapProvider().delete(sandbox_id="a/b")
