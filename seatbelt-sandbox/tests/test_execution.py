"""Live `sandbox-exec` tests: run real commands under the Seatbelt profile.

Require macOS + `/usr/bin/sandbox-exec`; skipped elsewhere. Cover behavior
only verifiable by running a sandboxed process: the `/Users` read fence,
`read_paths` opt-in, the `/tmp`->`/private/tmp` resolution, env scrubbing,
profile location outside the launch dir, the network default, and process
bootstrap.

Each test uses a throwaway `sandbox` fixture; `delete()` only unlinks the
profile file. Fence probes use the ``outside_dir`` fixture (an ad hoc dir
under ``$HOME``).
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
from dcode_seatbelt_sandbox.provider import (
    _PROFILES_ROOT,
    SeatbeltProvider,
    SeatbeltSandbox,
)

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("sandbox-exec") is None,
    reason="seatbelt tests require macOS sandbox-exec",
)

# Seatbelt denies reads under /Users at the syscall level, so a fenced host
# path surfaces as EPERM ("Operation not permitted").
FENCE_ERR = "Operation not permitted"


@pytest.fixture
def provider() -> SeatbeltProvider:
    return SeatbeltProvider()


@pytest.fixture
def sandbox(provider: SeatbeltProvider, request):
    """A throwaway sandbox. `read_paths` can be set via indirect parametrization."""
    read_paths = getattr(request, "param", None)
    sb = _make_sandbox(provider, request.node.name, read_paths)
    yield sb
    with contextlib.suppress(Exception):
        provider.delete(sandbox_id=sb.id)


@pytest.fixture
def sandbox_with_test_paths(
    provider: SeatbeltProvider, request, outside_dir: Path
) -> Iterator[SeatbeltSandbox]:
    """A throwaway sandbox with `outside_dir` bound as a read_path."""
    sb = _make_sandbox(provider, request.node.name, [str(outside_dir)])
    yield sb
    with contextlib.suppress(Exception):
        provider.delete(sandbox_id=sb.id)


def _make_sandbox(
    provider: SeatbeltProvider, node_name: str, read_paths: list[str] | None
) -> SeatbeltSandbox:
    # Sanitize the node name to the [A-Za-z0-9._-] charset the id validator
    # requires (parametrize adds "[None]" / "[sandbox0]" suffixes).
    raw = f"{os.getpid()}-{node_name}"
    safe = "".join(c if c.isalnum() or c in "._-" else "-" for c in raw)
    return provider.get_or_create(sandbox_id=f"pytest-{safe}", read_paths=read_paths)


# the home fence


class TestHomeFence:
    """By default, all of /Users is denied except the launch dir. `read_paths`
    re-allows the listed subpaths."""

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_home_denied_by_default(
        self, sandbox: SeatbeltSandbox, outside_dir: Path
    ) -> None:
        target = outside_dir / "marker.txt"
        target.write_text("fence-probe")
        r = sandbox.execute(f"cat {shlex.quote(str(target))} 2>&1; echo exit=$?")
        assert FENCE_ERR in r.output, r.output


# launch dir read/write


class TestLaunchDir:
    """The launch dir is the read/write area with zero config."""

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_launch_dir_readable(self, sandbox: SeatbeltSandbox) -> None:
        launch = sandbox._launch
        probe = launch / "pyproject.toml"
        if not probe.is_file():
            pytest.skip(f"no probe file at {probe}")
        r = sandbox.execute(f"head -c 1 {shlex.quote(str(probe))} 2>&1")
        assert r.exit_code == 0, r.output
        assert FENCE_ERR not in r.output, r.output

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_launch_dir_writable(self, sandbox: SeatbeltSandbox) -> None:
        launch = sandbox._launch
        marker = f"seatbelt-wtest-{os.getpid()}"
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
        self, sandbox: SeatbeltSandbox, outside_dir: Path
    ) -> None:
        r = sandbox.execute(f"ls {shlex.quote(str(outside_dir))} 2>&1; echo exit=$?")
        assert FENCE_ERR in r.output, r.output


# read_paths opt-in


class TestReadPaths:
    """`read_paths` adds extra readable /Users subpaths."""

    def test_read_paths_allows_extra_dir(
        self,
        sandbox_with_test_paths: SeatbeltSandbox,
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
        sandbox_with_test_paths: SeatbeltSandbox,
        outside_dir: Path,
    ) -> None:
        # read_paths re-allows reads only; writes outside the launch dir stay denied.
        target = outside_dir / f"rwtest-{os.getpid()}"
        try:
            r = sandbox_with_test_paths.execute(
                f"echo x > {target} 2>&1; echo exit=$?"
            )
            assert FENCE_ERR in r.output, r.output
        finally:
            with contextlib.suppress(FileNotFoundError):
                target.unlink()


# process bootstrap


class TestProcessBootstrap:
    """A sandboxed process must actually run (broad reads let dyld bootstrap)."""

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_echo_runs(self, sandbox: SeatbeltSandbox) -> None:
        r = sandbox.execute("echo hello-from-seatbelt")
        assert r.exit_code == 0, f"execute failed: {r.output!r}"
        assert "hello-from-seatbelt" in r.output

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_pwd_reports_launch_dir(self, sandbox: SeatbeltSandbox) -> None:
        r = sandbox.execute("pwd")
        launch = sandbox._launch
        assert str(launch) in r.output or str(launch.resolve()) in r.output, r.output

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_concurrent_execute_succeeds(self, sandbox: SeatbeltSandbox) -> None:
        """Parallel tool calls on one sandbox must all run, not fail.

        dcode fans out independent tool calls concurrently on a single sandbox.
        If the provider rewrites the shared profile file on every call, a
        `sandbox-exec -f` reader can observe a truncated file and fail before
        the sandboxed process starts. Every concurrent call must succeed.
        """
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
        self, sandbox: SeatbeltSandbox, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # monkeypatch scopes the keys to this test even under shared
        # pytest-xdist workers; never leak into other tests.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-SHOULD-NOT-LEAK")
        monkeypatch.setenv("GITHUB_TOKEN", "ghp-SHOULD-NOT-LEAK")
        r = sandbox.execute("printenv | sort")
        assert "SHOULD-NOT-LEAK" not in r.output, "secret leaked into sandbox"
        assert "OPENAI_API_KEY" not in r.output
        assert "GITHUB_TOKEN" not in r.output

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_minimal_env_present(self, sandbox: SeatbeltSandbox) -> None:
        r = sandbox.execute("printenv | sort")
        for key in ("PATH", "HOME", "TMPDIR", "SHELL", "LANG"):
            assert key in r.output, f"{key} missing from sandbox env"


# profile generation (seatbelt-specific: SBPL profile on disk)


class TestProfile:
    """The SBPL profile must live outside the launch dir, use the resolved
    /private/tmp path, contain the broad reads + /Users deny + launch-dir
    re-allow, and narrow /dev writes to /dev/null."""

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_profile_not_in_launch_dir(self, sandbox: SeatbeltSandbox) -> None:
        profile = sandbox._write_profile()
        assert profile.exists(), f"profile not at {profile}"
        assert _PROFILES_ROOT != sandbox._launch
        assert _PROFILES_ROOT not in sandbox._launch.parents

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_profile_stored_in_profiles_root(self, sandbox: SeatbeltSandbox) -> None:
        profile = sandbox._write_profile()
        assert profile.exists(), f"profile not at {profile}"

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_profile_uses_resolved_launch_dir(self, sandbox: SeatbeltSandbox) -> None:
        profile = sandbox._write_profile()
        text = profile.read_text()
        # The SBPL subpath literal must be the resolved (/private/tmp-aware)
        # path or write rules silently fail.
        resolved = str(sandbox._launch.resolve())
        assert f'subpath "{resolved}"' in text, text

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_profile_contains_broad_reads_and_users_deny(
        self, sandbox: SeatbeltSandbox
    ) -> None:
        profile = sandbox._write_profile()
        text = profile.read_text()
        assert "(allow file-read*)" in text
        assert "(allow file-read-data)" in text
        assert "(allow file-read-metadata)" in text
        assert '(deny file-read*\n    (subpath "/Users"))' in text
        assert '(deny file-read-data\n    (subpath "/Users"))' in text
        assert '(deny file-read-metadata\n    (subpath "/Users"))' in text

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_profile_narrows_dev_writes(self, sandbox: SeatbeltSandbox) -> None:
        profile = sandbox._write_profile()
        text = profile.read_text()
        assert '(literal "/dev/null")' in text
        assert '(subpath "/dev")' not in text.replace('(literal "/dev/null")', "")

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_profile_network_rule_default_allow(self, sandbox: SeatbeltSandbox) -> None:
        profile = sandbox._write_profile()
        text = profile.read_text()
        assert "(allow network*)" in text

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_profile_network_rule_opt_out(self, sandbox: SeatbeltSandbox) -> None:
        sandbox._network = False
        # Probe `_profile_text` directly: the on-disk profile reflects the
        # construction-time network=True, so it can't test the opt-out rule.
        text = sandbox._profile_text()
        assert "(deny network*)" in text

    def test_delete_removes_profile_file(
        self, provider: SeatbeltProvider, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        # Pid-namespace the id so concurrent pytest-xdist runs don't collide.
        sid = f"pytest-cleanup-profile-{os.getpid()}"
        sb = provider.get_or_create(sandbox_id=sid)
        sb._write_profile()
        profile = _PROFILES_ROOT / f"{sid}.sb"
        assert profile.exists()
        provider.delete(sandbox_id=sid)
        assert not profile.exists()


# grep (BSD grep -Z vs --null)


class TestGrep:
    """`SeatbeltSandbox` overrides `grep`/`agrep` to use `--null` instead of
    `grep -rHnFZ`: BSD grep (macOS) treats `-Z` as `--decompress` and emits
    plain `path:line:text`, which the base parser can't split. These run the
    override through real `sandbox-exec` + BSD grep and assert it returns
    matches, not an error.
    """

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_grep_returns_matches_not_error(self, sandbox: SeatbeltSandbox) -> None:
        launch = sandbox._launch
        target = launch / f"seatbelt-greptest-{os.getpid()}"
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
    def test_grep_basename_glob(self, sandbox: SeatbeltSandbox) -> None:
        launch = sandbox._launch
        target = launch / f"seatbelt-grepglob-{os.getpid()}"
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
    def test_grep_max_count_truncates(self, sandbox: SeatbeltSandbox) -> None:
        launch = sandbox._launch
        target = launch / f"seatbelt-grepcap-{os.getpid()}"
        target.mkdir(exist_ok=True)
        try:
            # Two files, two matches each -> 4 total; cap at 1, so the builder
            # reads cap+1 and the parser flags truncation.
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
        self, sandbox: SeatbeltSandbox
    ) -> None:
        launch = sandbox._launch
        target = launch / f"seatbelt-grepnomatch-{os.getpid()}"
        target.mkdir(exist_ok=True)
        try:
            (target / "a.txt").write_text("nothing relevant\n")
            result = sandbox.grep("absent-pattern", path=str(target))
            assert result.error is None, result.error
            assert result.matches == []
        finally:
            shutil.rmtree(target, ignore_errors=True)

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_grep_slash_glob(self, sandbox: SeatbeltSandbox) -> None:
        # A `/`-containing glob routes to the in-process Python template (the
        # basename-only `--include` path can't express path-relative globs).
        launch = sandbox._launch
        target = launch / f"seatbelt-grepslash-{os.getpid()}"
        target.mkdir(exist_ok=True)
        try:
            (target / "skip.txt").write_text("needle here\n")
            (target / "pkg").mkdir()
            (target / "pkg" / "deep.py").write_text("needle\n")
            result = sandbox.grep("needle", path=str(target), glob="pkg/*.py")
            assert result.error is None, result.error
            assert result.matches is not None, result.matches
            assert len(result.matches) == 1, result.matches
            assert result.matches[0]["path"].endswith("deep.py")
        finally:
            shutil.rmtree(target, ignore_errors=True)


# file operations through the sandbox (execute-routed transport)


class TestFileOps:
    """Upload/download round-trip, and that out-of-launch-dir paths are denied
    by the profile (not by us)."""

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_upload_download_roundtrip(self, sandbox: SeatbeltSandbox) -> None:
        launch = sandbox._launch
        path = str(launch / f"seatbelt-roundtrip-{os.getpid()}.txt")
        try:
            up = sandbox.upload_files([(path, b"seatbelt-roundtrip")])
            assert up[0].error is None, up[0].error
            dl = sandbox.download_files([path])
            assert dl[0].content == b"seatbelt-roundtrip"
            assert dl[0].error is None
        finally:
            with contextlib.suppress(FileNotFoundError):
                Path(path).unlink()

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_download_missing_file_returns_file_not_found(
        self, sandbox: SeatbeltSandbox
    ) -> None:
        launch = sandbox._launch
        path = str(launch / f"nonexistent-{os.getpid()}.txt")
        results = sandbox.download_files([path])
        assert len(results) == 1
        assert results[0].content is None
        assert results[0].error == "file_not_found"

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_out_of_launch_dir_download_denied(
        self, sandbox: SeatbeltSandbox, outside_dir: Path
    ) -> None:
        # The outside-dir file EXISTS on the host, so a pass can't be explained
        # by file_not_found: the fence must surface as a read denial.
        target = outside_dir / "fence-probe.txt"
        target.write_text("fence-probe")
        results = sandbox.download_files([str(target)])
        assert results[0].content is None
        assert results[0].error is not None
        assert results[0].error != "file_not_found", results[0].error

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_out_of_launch_dir_upload_denied(
        self, sandbox: SeatbeltSandbox, outside_dir: Path
    ) -> None:
        target = outside_dir / "pwned"
        results = sandbox.upload_files([(str(target), b"x")])
        assert results[0].error is not None


# write confinement


class TestWriteConfinement:
    """Writes outside the launch dir must be blocked by the Seatbelt profile."""

    @pytest.mark.parametrize("sandbox", [None], indirect=True)
    def test_cannot_write_to_home_outside_launch(
        self, sandbox: SeatbeltSandbox, outside_dir: Path
    ) -> None:
        target = outside_dir / f"wtest-{os.getpid()}"
        try:
            r = sandbox.execute(f"echo pwned > {target} 2>&1; echo exit=$?")
            assert FENCE_ERR in r.output, r.output
        finally:
            with contextlib.suppress(FileNotFoundError):
                target.unlink()
