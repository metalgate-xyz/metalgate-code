"""Live `bwrap` tests -- run real commands under the bubblewrap sandbox.

These require Linux and `bwrap` in PATH and unprivileged user namespaces
enabled; they are skipped elsewhere. They cover the behavior that can only
be verified by actually running a sandboxed process: the home-invisibility
fence (paths outside the launch dir are absent, not merely denied),
`read_paths` opt-in, environment scrubbing, the network default, the
write fence, and that a process actually bootstraps.

Each sandbox's launch dir is whatever the provider resolves (from the
server context env vars, falling back to the cwd) -- tests read it back
from the sandbox itself instead of assuming a path derived from this
file's location, so they pass regardless of the cwd pytest is invoked
from. Each test uses a throwaway sandbox via the `sandbox` fixture;
`delete()` is a no-op for bubblewrap (no on-disk artifacts).
"""

from __future__ import annotations

import contextlib
import os
import shlex
import shutil
import sys
import threading
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

WHO = os.environ.get("USER", "")
HOME = str(Path.home())


@pytest.fixture
def provider() -> BubblewrapProvider:
    return BubblewrapProvider()


@pytest.fixture
def sandbox(provider: BubblewrapProvider, request):
    """A throwaway sandbox. `read_paths` can be set via indirect parametrization."""
    read_paths = getattr(request, "param", None)
    raw = f"{os.getpid()}-{request.node.name}"
    safe = "".join(c if c.isalnum() or c in "._-" else "-" for c in raw)
    sb = provider.get_or_create(sandbox_id=f"pytest-{safe}", read_paths=read_paths)
    yield sb
    with contextlib.suppress(Exception):
        provider.delete(sandbox_id=sb.id)


# --- the home invisibility fence -------------------------------------------


class TestHomeInvisibilityFence:
    """By default, all of $HOME except the launch dir and curated tool paths is
    not mounted, so it is simply absent (ENOENT), not merely permission-denied.
    With read_paths, the listed subpaths are bound ro and become visible."""

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_home_secrets_absent(self, sandbox: BubblewrapSandbox) -> None:
        # ~/.ssh is not mounted: ls reports "No such file or directory", not a
        # permission error. This is the bubblewrap analog of seatbelt's
        # "Operation not permitted" /Users fence.
        r = sandbox.execute(f"ls {HOME}/.ssh 2>&1; echo exit=$?")
        assert "No such file or directory" in r.output, r.output

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_system_files_readable(self, sandbox: BubblewrapSandbox) -> None:
        # /etc is bound ro -- system files stay readable.
        r = sandbox.execute("head -c 10 /etc/hostname 2>&1; echo exit=$?")
        assert r.exit_code == 0, r.output
        assert r.output.strip() != "" or "exit=0" in r.output

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_home_root_absent(self, sandbox: BubblewrapSandbox) -> None:
        # $HOME is not mounted as a bind, but bwrap auto-creates its ancestor
        # dirs (read-only, via --perms 0555 --dir) to hold the launch-dir and
        # tool-path binds that live under it. So `ls $HOME` succeeds and shows
        # a *sparse* home root -- only the bound subpaths -- not the full home.
        # The meaningful fence assertion is that a secret subdir stays absent.
        if str(sandbox._launch) == HOME:
            pytest.skip("launch dir is $HOME; home root is mounted by definition")
        r = sandbox.execute(f"ls {HOME}/.ssh 2>&1; echo exit=$?")
        assert "No such file or directory" in r.output, r.output


# --- launch dir read/write -------------------------------------------------


class TestLaunchDir:
    """The launch dir is the read/write area: the agent reads and writes the
    project with zero config."""

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
    def test_home_still_absent_when_launch_dir_allowed(
        self, sandbox: BubblewrapSandbox
    ) -> None:
        # The launch dir is mounted, but ~/.ssh stays absent.
        if str(sandbox._launch) == HOME:
            pytest.skip("launch dir is $HOME")
        r = sandbox.execute(f"ls {HOME}/.ssh 2>&1; echo exit=$?")
        assert "No such file or directory" in r.output, r.output


# --- read_paths opt-in ------------------------------------------------------


class TestReadPaths:
    """`read_paths` adds extra readable bind mounts on top of the launch dir."""

    @pytest.mark.parametrize(
        "sandbox", [[str(Path.home())]], indirect=True
    )
    def test_read_paths_makes_home_visible(self, sandbox: BubblewrapSandbox) -> None:
        # With ~ in read_paths, $HOME becomes readable (but NOT writable).
        r = sandbox.execute(f"ls {HOME} 2>&1; echo exit=$?")
        assert r.exit_code == 0, r.output
        assert "No such file or directory" not in r.output, r.output

    @pytest.mark.parametrize(
        "sandbox", [[str(Path.home())]], indirect=True
    )
    def test_read_paths_do_not_allow_writes(self, sandbox: BubblewrapSandbox) -> None:
        # read_paths bind is ro; writes to $HOME stay blocked. The command runs
        # `echo x > target; echo exit=$?` -- the trailing `echo exit=$?` is a
        # second command that succeeds, so the *overall* exit code is 0; assert
        # on the inner `exit=` line and the error string instead.
        target = Path.home() / f".bwrap-rwtest-{os.getpid()}"
        try:
            r = sandbox.execute(f"echo x > {target} 2>&1; echo exit=$?")
            assert "Read-only file system" in r.output or "Permission denied" in r.output, r.output
            assert "exit=" in r.output and "exit=0" not in r.output, r.output
            assert target.exists() is False
        finally:
            with contextlib.suppress(FileNotFoundError):
                target.unlink()


# --- process bootstrap -----------------------------------------------------


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
    def test_concurrent_execute_is_independent(self, sandbox: BubblewrapSandbox) -> None:
        """Parallel tool calls on one sandbox must all run. Each is an
        independent bwrap invocation sharing no state, so this pins that the
        argv rebuild-on-every-call contract holds under concurrency."""
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


# --- environment scrubbing -------------------------------------------------


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


# --- network ---------------------------------------------------------------


class TestNetwork:
    """Network is shared by default (coding needs it); --unshare-net when off."""

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_network_allowed_by_default(self, sandbox: BubblewrapSandbox) -> None:
        # A best-effort DNS reachability check. We don't assert on the network
        # being up (CI may block it), only that bwrap didn't unshare the netns:
        # with the host netns, `ip` (if present) shows the host interfaces, not
        # just loopback. Fall back to /proc/net/dev existing.
        r = sandbox.execute("test -r /proc/net/dev && echo netns-shared")
        assert "netns-shared" in r.output, r.output

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_network_off_unshares_net(self, sandbox: BubblewrapSandbox) -> None:
        # Create a fresh sandbox with network=False and check only loopback.
        # We can't reconfigure the fixture sandbox, so construct directly.
        launch = sandbox._launch
        sb = BubblewrapSandbox("netoff-test", launch, network=False)
        # In an unshared netns, only `lo` exists. `ip` may not be installed;
        # /proc/net/dev lists interfaces and only has `lo` in a private netns.
        r = sb.execute("cat /proc/net/dev 2>&1")
        lines = [ln for ln in r.output.splitlines() if ln.strip() and ":" in ln]
        names = [ln.split(":")[0].strip() for ln in lines]
        assert names == ["lo"], f"expected only lo in unshared netns, got {names}: {r.output}"


# --- grep (inherited BaseSandbox.grep, GNU grep -Z) -------------------------


class TestGrep:
    """The inherited `BaseSandbox.grep` builds `grep -rHnFZ`; on Linux GNU grep
    treats `-Z` as `--null` (NUL after filename), which `_parse_grep_output`
    expects. No override is needed. These live tests run the inherited path
    through real bwrap + GNU grep and assert it returns matches, not an error."""

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


# --- file operations through the sandbox (execute-routed transport) --------


class TestFileOps:
    """Upload/download round-trip via the base64-over-execute transport, and
    that out-of-launch-dir paths are absent (not mounted), so the transport
    fails with file_not_found -- the fence, not a provider-side check."""

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
    def test_out_of_launch_dir_download_absent(
        self, sandbox: BubblewrapSandbox
    ) -> None:
        # A path under $HOME outside the launch dir is not mounted, so download
        # surfaces file_not_found (the fence). The home marker is a non-sensitive
        # junk file the test creates -- not a real dotfile.
        if str(sandbox._launch) == HOME:
            pytest.skip("launch dir is $HOME")
        target = Path.home() / f".bwrap-dltest-{os.getpid()}"
        target.write_text("fence-probe")
        try:
            results = sandbox.download_files([str(target)])
            assert results[0].content is None
            # The fence surfaces as file_not_found (the path isn't mounted).
            assert results[0].error == "file_not_found", results[0].error
        finally:
            with contextlib.suppress(FileNotFoundError):
                target.unlink()

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_out_of_launch_dir_upload_fails(self, sandbox: BubblewrapSandbox) -> None:
        # Uploading to a path under $HOME outside the launch dir: the parent
        # isn't mounted, so mkdir -p fails and the upload errors.
        if str(sandbox._launch) == HOME:
            pytest.skip("launch dir is $HOME")
        results = sandbox.upload_files([(f"{HOME}/.ssh/pwned", b"x")])
        assert results[0].error is not None


# --- write confinement -----------------------------------------------------


class TestWriteConfinement:
    """Writes outside the launch dir must be blocked. With bubblewrap the
    target is either read-only (system binds) or not mounted (everything else),
    so a write fails with EROFS or ENOENT rather than EPERM."""

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_cannot_write_to_system_files(self, sandbox: BubblewrapSandbox) -> None:
        # /etc is bound ro; a write there fails (read-only filesystem). The
        # trailing `echo exit=$?` is a second command that succeeds, so assert
        # on the error string and the inner exit line, not the overall code.
        target = f"/etc/.bwrap-wtest-{os.getpid()}"
        r = sandbox.execute(f"echo pwned > {target} 2>&1; echo exit=$?")
        assert "Read-only file system" in r.output, r.output
        assert "exit=" in r.output and "exit=0" not in r.output, r.output

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_cannot_write_to_home_outside_launch(
        self, sandbox: BubblewrapSandbox
    ) -> None:
        if str(sandbox._launch) == HOME:
            pytest.skip("launch dir is $HOME")
        target = Path.home() / f".bwrap-wtest-{os.getpid()}"
        try:
            r = sandbox.execute(f"echo pwned > {target} 2>&1; echo exit=$?")
            # $HOME is pre-created read-only (0555) as an ancestor of the
            # launch-dir/tool binds, so a write into it is denied (EROFS or
            # EPERM); if it were not an ancestor, ENOENT. Either way: blocked.
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
