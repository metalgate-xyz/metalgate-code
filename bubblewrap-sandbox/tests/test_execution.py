"""Live `bwrap` tests: run real commands under the bubblewrap sandbox.

Require Linux + `bwrap` + unprivileged user namespaces; skipped elsewhere.
Cover behavior only verifiable by running a sandboxed process: the
home-invisibility fence, `read_paths` opt-in, env scrubbing, the network
default, the write fence, and process bootstrap.

Each test uses a throwaway `sandbox` fixture; `delete()` is a no-op for
bubblewrap. Fence probes use the ``outside_dir`` fixture (an ad hoc dir under
``$HOME``).
"""

from __future__ import annotations

import contextlib
import os
import shlex
import shutil
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from dcode_bubblewrap_sandbox.provider import (
    BubblewrapProvider,
    BubblewrapSandbox,
)

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("bwrap") is None,
    reason="bubblewrap tests require Linux bwrap with unprivileged user namespaces",
)

# Bubblewrap does not mount $HOME (except the launch dir and curated tool
# paths), so a fenced host path is ENOENT inside the sandbox.
FENCE_ERR = "No such file or directory"


@pytest.fixture
def provider() -> BubblewrapProvider:
    return BubblewrapProvider()


@pytest.fixture
def sandbox(provider: BubblewrapProvider, request):
    """A throwaway sandbox. `read_paths` can be set via indirect parametrization."""
    read_paths = getattr(request, "param", None)
    sb = _make_sandbox(provider, request.node.name, read_paths)
    yield sb
    with contextlib.suppress(Exception):
        provider.delete(sandbox_id=sb.id)


@pytest.fixture
def sandbox_with_test_paths(
    provider: BubblewrapProvider, request, outside_dir: Path
) -> Iterator[BubblewrapSandbox]:
    """A throwaway sandbox with `outside_dir` bound as a read_path."""
    sb = _make_sandbox(provider, request.node.name, [str(outside_dir)])
    yield sb
    with contextlib.suppress(Exception):
        provider.delete(sandbox_id=sb.id)


def _make_sandbox(
    provider: BubblewrapProvider, node_name: str, read_paths: list[str] | None
) -> BubblewrapSandbox:
    raw = f"{os.getpid()}-{node_name}"
    safe = "".join(c if c.isalnum() or c in "._-" else "-" for c in raw)
    return provider.get_or_create(sandbox_id=f"pytest-{safe}", read_paths=read_paths)


# the home fence


class TestHomeFence:
    """By default, $HOME except the launch dir and tool paths is unmounted
    (ENOENT). `read_paths` binds listed subpaths read-only."""

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_home_denied_by_default(
        self, sandbox: BubblewrapSandbox, outside_dir: Path
    ) -> None:
        target = outside_dir / "marker.txt"
        target.write_text("fence-probe")
        r = sandbox.execute(f"cat {shlex.quote(str(target))} 2>&1; echo exit=$?")
        assert FENCE_ERR in r.output, r.output


# launch dir read/write


class TestLaunchDir:
    """The launch dir is the read/write area with zero config."""

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_launch_dir_readable(self, sandbox: BubblewrapSandbox) -> None:
        launch = sandbox._launch
        probe = launch / "pyproject.toml"
        if not probe.is_file():
            pytest.skip(f"no probe file at {probe}")
        r = sandbox.execute(f"head -c 1 {shlex.quote(str(probe))} 2>&1")
        assert r.exit_code == 0, r.output

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_launch_dir_writable(self, sandbox: BubblewrapSandbox) -> None:
        launch = sandbox._launch
        marker = f"bwrap-wtest-{os.getpid()}"
        target = launch / marker
        try:
            r = sandbox.execute(f"echo data > {target} 2>&1; echo exit=$?")
            assert r.exit_code == 0, r.output
            assert target.read_text().strip() == "data"
        finally:
            with contextlib.suppress(FileNotFoundError):
                target.unlink()

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_home_still_denied_when_launch_dir_allowed(
        self, sandbox: BubblewrapSandbox, outside_dir: Path
    ) -> None:
        r = sandbox.execute(f"ls {shlex.quote(str(outside_dir))} 2>&1; echo exit=$?")
        assert FENCE_ERR in r.output, r.output


# read_paths opt-in


class TestReadPaths:
    """`read_paths` adds extra readable ro bind mounts."""

    def test_read_paths_allows_extra_dir(
        self,
        sandbox_with_test_paths: BubblewrapSandbox,
        outside_dir: Path,
    ) -> None:
        # With the outside dir in read_paths, it becomes readable.
        r = sandbox_with_test_paths.execute(
            f"ls {shlex.quote(str(outside_dir))} 2>&1; echo exit=$?"
        )
        assert r.exit_code == 0, r.output
        assert FENCE_ERR not in r.output, r.output

    def test_read_paths_do_not_allow_writes(
        self,
        sandbox_with_test_paths: BubblewrapSandbox,
        outside_dir: Path,
    ) -> None:
        # The read_paths bind is ro. The trailing `echo exit=$?` is a second
        # command that succeeds, so assert on the inner exit= line, not the
        # overall exit code.
        target = outside_dir / f"rwtest-{os.getpid()}"
        try:
            r = sandbox_with_test_paths.execute(
                f"echo x > {target} 2>&1; echo exit=$?"
            )
            assert (
                "Read-only file system" in r.output or "Permission denied" in r.output
            ), r.output
            assert "exit=" in r.output and "exit=0" not in r.output, r.output
            assert target.exists() is False
        finally:
            with contextlib.suppress(FileNotFoundError):
                target.unlink()


# process bootstrap


class TestProcessBootstrap:
    """A sandboxed process must actually run (system ro-binds let it bootstrap)."""

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_echo_runs(self, sandbox: BubblewrapSandbox) -> None:
        r = sandbox.execute("echo hello-from-bwrap")
        assert r.exit_code == 0, f"execute failed: {r.output!r}"
        assert "hello-from-bwrap" in r.output

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_pwd_reports_launch_dir(self, sandbox: BubblewrapSandbox) -> None:
        r = sandbox.execute("pwd")
        launch = sandbox._launch
        assert str(launch) in r.output or str(launch.resolve()) in r.output, r.output

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_concurrent_execute_succeeds(self, sandbox: BubblewrapSandbox) -> None:
        """Parallel tool calls on one sandbox must all run. Each is an
        independent bwrap invocation sharing no state."""
        n = 16
        results: list = [None] * n
        errs: list = [None] * n

        def run(i: int) -> None:
            try:
                results[i] = sandbox.execute(f"echo call-{i}")
            except Exception as exc:  # noqa: BLE001
                errs[i] = exc

        threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for i, r in enumerate(results):
            assert errs[i] is None, f"call {i} raised: {errs[i]!r}"
            assert r is not None, f"call {i} produced no result"
            assert r.exit_code == 0, f"call {i} failed: {r.output!r}"
            assert f"call-{i}" in r.output, f"call {i}: {r.output!r}"


# environment scrubbing


class TestEnvironmentScrubbing:
    """dcode's environment (with secrets) must NOT be inherited."""

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_secrets_not_inherited(
        self, sandbox: BubblewrapSandbox, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-SHOULD-NOT-LEAK")
        monkeypatch.setenv("GITHUB_TOKEN", "ghp-SHOULD-NOT-LEAK")
        r = sandbox.execute("printenv | sort")
        assert "SHOULD-NOT-LEAK" not in r.output, "secret leaked into sandbox"
        assert "OPENAI_API_KEY" not in r.output
        assert "GITHUB_TOKEN" not in r.output

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_minimal_env_present(self, sandbox: BubblewrapSandbox) -> None:
        r = sandbox.execute("printenv | sort")
        for key in ("PATH", "HOME", "TMPDIR", "SHELL", "LANG"):
            assert key in r.output, f"{key} missing from sandbox env"


# network (bubblewrap-specific: live netns check)


class TestNetwork:
    """Network shared by default (coding needs it); --unshare-net when off."""

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_network_allowed_by_default(self, sandbox: BubblewrapSandbox) -> None:
        # Assert bwrap didn't unshare the netns: /proc/net/dev exists with the
        # host interfaces (not just loopback).
        r = sandbox.execute("test -r /proc/net/dev && echo netns-shared")
        assert "netns-shared" in r.output, r.output

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_network_off_unshares_net(self, sandbox: BubblewrapSandbox) -> None:
        # Can't reconfigure the fixture sandbox, so construct directly.
        launch = sandbox._launch
        sb = BubblewrapSandbox("netoff-test", launch, network=False)
        # In an unshared netns, /proc/net/dev lists only `lo`.
        r = sb.execute("cat /proc/net/dev 2>&1")
        lines = [ln for ln in r.output.splitlines() if ln.strip() and ":" in ln]
        names = [ln.split(":")[0].strip() for ln in lines]
        assert names == ["lo"], (
            f"expected only lo in unshared netns, got {names}: {r.output}"
        )


# grep (inherited BaseSandbox.grep, GNU grep -Z)


class TestGrep:
    """Inherited `BaseSandbox.grep` builds `grep -rHnFZ`; GNU grep treats `-Z`
    as `--null` (NUL after filename), which `_parse_grep_output` expects. No
    override needed: these run the inherited path through real bwrap + GNU
    grep and assert it returns matches, not an error."""

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_grep_returns_matches_not_error(self, sandbox: BubblewrapSandbox) -> None:
        launch = sandbox._launch
        target = launch / f"bwrap-greptest-{os.getpid()}"
        target.mkdir(exist_ok=True)
        try:
            (target / "a.txt").write_text("hello world\nfoo bar\n")
            (target / "b.txt").write_text("goodbye\nhello again\n")
            result = sandbox.grep("hello", path=str(target))
            assert result.error is None, result.error
            assert result.matches is not None, "expected at least one match"
            paths = {m["path"] for m in result.matches}
            assert any(p.endswith("a.txt") for p in paths), result.matches
            assert any(p.endswith("b.txt") for p in paths), result.matches
            texts = {m["text"] for m in result.matches}
            assert "hello world" in texts
            assert "hello again" in texts
        finally:
            shutil.rmtree(target, ignore_errors=True)

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_grep_basename_glob(self, sandbox: BubblewrapSandbox) -> None:
        launch = sandbox._launch
        target = launch / f"bwrap-grepglob-{os.getpid()}"
        target.mkdir(exist_ok=True)
        try:
            (target / "match.py").write_text("needle here\n")
            (target / "skip.txt").write_text("needle here\n")
            result = sandbox.grep("needle", path=str(target), glob="*.py")
            assert result.error is None, result.error
            assert result.matches is not None, result.matches
            assert len(result.matches) == 1, result.matches
            assert result.matches[0]["path"].endswith("match.py")
        finally:
            shutil.rmtree(target, ignore_errors=True)

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_grep_max_count_truncates(self, sandbox: BubblewrapSandbox) -> None:
        launch = sandbox._launch
        target = launch / f"bwrap-grepcap-{os.getpid()}"
        target.mkdir(exist_ok=True)
        try:
            (target / "a.txt").write_text("needle\nneedle\n")
            (target / "b.txt").write_text("needle\nneedle\n")
            result = sandbox.grep("needle", path=str(target), max_count=1)
            assert result.error is None, result.error
            assert result.matches is not None, result.matches
            assert len(result.matches) == 1, result.matches
            assert result.truncated is True
        finally:
            shutil.rmtree(target, ignore_errors=True)

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_grep_no_match_returns_empty_not_error(
        self, sandbox: BubblewrapSandbox
    ) -> None:
        launch = sandbox._launch
        target = launch / f"bwrap-grepnomatch-{os.getpid()}"
        target.mkdir(exist_ok=True)
        try:
            (target / "a.txt").write_text("nothing relevant\n")
            result = sandbox.grep("absent-pattern", path=str(target))
            assert result.error is None, result.error
            assert result.matches == []
        finally:
            shutil.rmtree(target, ignore_errors=True)


# file operations through the sandbox (execute-routed transport)


class TestFileOps:
    """Upload/download round-trip, and that out-of-launch-dir paths are
    unmounted so the transport fails with file_not_found (the fence)."""

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_upload_download_roundtrip(self, sandbox: BubblewrapSandbox) -> None:
        launch = sandbox._launch
        path = str(launch / f"bwrap-roundtrip-{os.getpid()}.txt")
        try:
            up = sandbox.upload_files([(path, b"bwrap-roundtrip")])
            assert up[0].error is None, up[0].error
            dl = sandbox.download_files([path])
            assert dl[0].content == b"bwrap-roundtrip"
            assert dl[0].error is None
        finally:
            with contextlib.suppress(FileNotFoundError):
                Path(path).unlink()

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_download_missing_file_returns_file_not_found(
        self, sandbox: BubblewrapSandbox
    ) -> None:
        launch = sandbox._launch
        path = str(launch / f"nonexistent-{os.getpid()}.txt")
        results = sandbox.download_files([path])
        assert len(results) == 1
        assert results[0].content is None
        assert results[0].error == "file_not_found"

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_out_of_launch_dir_download_blocked(
        self, sandbox: BubblewrapSandbox, outside_dir: Path
    ) -> None:
        target = outside_dir / "fence-probe.txt"
        target.write_text("fence-probe")
        results = sandbox.download_files([str(target)])
        assert results[0].content is None
        assert results[0].error == "file_not_found", results[0].error

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_out_of_launch_dir_upload_blocked(
        self, sandbox: BubblewrapSandbox, outside_dir: Path
    ) -> None:
        target = outside_dir / "pwned"
        results = sandbox.upload_files([(str(target), b"x")])
        assert results[0].error is not None


# write confinement


class TestWriteConfinement:
    """Writes outside the launch dir must be blocked: the target is ro
    (system binds) or unmounted (everything else), so writes fail with EROFS
    or ENOENT, not EPERM."""

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_cannot_write_to_home_outside_launch(
        self, sandbox: BubblewrapSandbox, outside_dir: Path
    ) -> None:
        target = outside_dir / f"wtest-{os.getpid()}"
        try:
            r = sandbox.execute(f"echo pwned > {target} 2>&1; echo exit=$?")
            # $HOME is not mounted or read-only, so the write is EROFS/EPERM,
            # or ENOENT if not an ancestor of a mounted bind. Either way: blocked.
            assert (
                "Read-only file system" in r.output
                or "Permission denied" in r.output
                or "No such file or directory" in r.output
            ), r.output
            assert "exit=" in r.output and "exit=0" not in r.output, r.output
            assert target.exists() is False
        finally:
            with contextlib.suppress(FileNotFoundError):
                target.unlink()
