"""Seatbelt provider tests: sandbox_id validation, SBPL read-allow block
builder, and the BaseSandbox abstract surface. Some cases exercise
`sandbox-exec`, so these require macOS.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from dcode_seatbelt_sandbox import SeatbeltSandbox
from dcode_seatbelt_sandbox.provider import (
    _ID_PATTERN,
    _ancestor_metadata_rules,
    _build_bsd_grep_cmd,
    _read_allow_block,
)

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="seatbelt tests require macOS",
)

# sandbox_id validation


class TestIdValidation:
    """`sandbox_id` only flows into a profile filename under `_PROFILES_ROOT`.
    The regex is a friendly-error fast-fail for filename safety, not a
    security boundary."""

    @pytest.mark.parametrize(
        "bad",
        [
            "a/b",  # path separator: subdirectory under _PROFILES_ROOT
            "a\\b",  # backslash (hazard on Windows-shared mounts)
            "",  # empty
            "foo bar",  # whitespace: breaks unquoted filenames
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
            "\x00",  # null byte: truncates C strings
        ],
    )
    def test_rejects_invalid_ids(self, bad: str) -> None:
        assert _ID_PATTERN.match(bad) is None

    @pytest.mark.parametrize(
        "ok",
        [
            "seatbelt-abc123",
            "my_project",
            "my.project",
            "a",
            "A1-B2.c3",
            "-n",  # leading dash: safe, id is never a CLI arg, only a filename
            ".hidden",  # leading dot: safe filename under _PROFILES_ROOT
            "..",  # two dots: passes charset; harmless as a filename (no dir to escape)
        ],
    )
    def test_accepts_valid_ids(self, ok: str) -> None:
        assert _ID_PATTERN.match(ok) is not None


# _read_allow_block (SBPL re-allow builder)


class TestReadAllowBlock:
    """`_read_allow_block` re-allows the launch dir + tool paths + read_paths
    under /Users; entries outside /Users are skipped (already covered by the
    broad allow)."""

    def test_launch_dir_emits_all_three_read_op_types(self) -> None:
        launch = "/Users/tester/launch"
        block = _read_allow_block(launch, [])
        assert f'(allow file-read* (subpath "{launch}"))' in block
        assert f'(allow file-read-data (subpath "{launch}"))' in block
        assert f'(allow file-read-metadata (subpath "{launch}"))' in block

    def test_read_paths_under_users_are_re_allowed(self) -> None:
        proj = "/Users/tester/code/proj"
        block = _read_allow_block("/Users/tester/launch", [proj])
        assert f'(allow file-read* (subpath "{proj}"))' in block
        assert f'(allow file-read-data (subpath "{proj}"))' in block
        assert f'(allow file-read-metadata (subpath "{proj}"))' in block

    def test_non_users_paths_are_skipped(self) -> None:
        # /usr/local and /opt are already readable via the broad allow and must
        # not appear in the re-allow block. The launch dir still appears.
        launch = "/Users/tester/launch"
        block = _read_allow_block(launch, ["/usr/local", "/opt/homebrew"])
        assert "/usr/local" not in block
        assert "/opt/homebrew" not in block
        assert f'subpath "{launch}"' in block

    def test_duplicates_are_deduped(self) -> None:
        launch = "/Users/tester/launch"
        block = _read_allow_block(launch, [launch, launch])
        # 3 op types, each appearing exactly once for the launch dir.
        assert block.count(f'subpath "{launch}"') == 3

    def test_paths_outside_users_yield_no_launch_reallow(self) -> None:
        # When the launch dir itself is outside /Users (e.g. under /tmp), it's
        # already covered by the broad read allow, so only /Users tool/config
        # paths appear.
        block = _read_allow_block("/private/tmp/launch", [])
        assert "/private/tmp/launch" not in block

    def test_paths_are_resolved_to_canonical_form(self, tmp_path: Path) -> None:
        """Symlinks/relative components are resolved so the SBPL `subpath`
        literal matches kernel I/O."""
        raw = str(tmp_path / "x" / ".." / "y")
        block = _read_allow_block("/Users/tester/launch", [raw])
        resolved = str((tmp_path / "y").resolve())
        if resolved.startswith("/Users/"):
            assert f'subpath "{resolved}"' in block
        else:
            # Not under /Users -> skipped; only the launch dir appears.
            assert f'subpath "{resolved}"' not in block


class TestAncestorMetadata:
    """`_ancestor_metadata_rules` emits metadata-only re-allow rules for /Users
    and each launch-dir ancestor, so `cd` can traverse without leaking
    contents."""

    def test_emits_users_and_each_ancestor(self) -> None:
        rules = _ancestor_metadata_rules("/Users/u/code/proj")
        assert '(allow file-read-metadata (subpath "/Users"))' in rules
        assert '(allow file-read-metadata (subpath "/Users/u"))' in rules
        assert '(allow file-read-metadata (subpath "/Users/u/code"))' in rules
        # The launch dir itself gets full read from _read_allow_block, not here.
        assert "/Users/u/code/proj" not in rules

    def test_metadata_only_no_data_or_file_read_star(self) -> None:
        rules = _ancestor_metadata_rules("/Users/u/code/proj")
        assert "file-read-data" not in rules
        assert "file-read*" not in rules
        assert "file-read-metadata" in rules

    def test_launch_outside_users_yields_empty(self) -> None:
        assert _ancestor_metadata_rules("/tmp/proj") == ""
        assert _ancestor_metadata_rules("/opt/proj") == ""


# BaseSandbox abstract surface


class TestBaseSandboxSurface:
    """Abstract members are implemented; derived methods come from BaseSandbox."""

    def test_sandbox_instantiates_and_has_abstract_members(
        self, tmp_path: Path
    ) -> None:
        sb = SeatbeltSandbox("surface-test", tmp_path)
        assert sb.id == "surface-test"
        for name in ("execute", "upload_files", "download_files"):
            assert callable(getattr(sb, name)), name

    def test_derived_methods_present(self, tmp_path: Path) -> None:
        sb = SeatbeltSandbox("surface-test", tmp_path)
        for name in ("ls", "read", "write", "edit", "delete", "grep", "glob"):
            assert callable(getattr(sb, name)), name


# BSD grep command builder


class TestBsdGrepBuilder:
    """`_build_bsd_grep_cmd` swaps the base `grep -rHnFZ` for `--null` so the
    output parses under BSD grep (macOS), where `-Z` is `--decompress`, not the
    NUL separator `_parse_grep_output` expects.

    These run without `sandbox-exec`: they assert the command shape and that
    the parser consumes real `--null`-shaped output.
    """

    def test_plain_search_uses_null_not_capital_z(self) -> None:
        cmd = _build_bsd_grep_cmd("hello", "/p", None)
        assert "-rHnF --null" in cmd
        assert "-rHnFZ" not in cmd

    def test_basename_glob_uses_null(self) -> None:
        cmd = _build_bsd_grep_cmd("hello", "/p", "*.py")
        assert "-rHnF --null" in cmd
        assert "--include='*.py'" in cmd
        assert "-rHnFZ" not in cmd

    def test_max_count_reads_one_past_cap(self) -> None:
        cmd = _build_bsd_grep_cmd("hello", "/p", None, max_count=10)
        assert "| head -n 11 || true" in cmd

    def test_slash_glob_uses_python_not_grep(self) -> None:
        # Slash-containing globs can't use `grep --include` (basename-only), so
        # they run an in-process Python search that writes its own
        # `path\0line:text` records.
        cmd = _build_bsd_grep_cmd("hello", "/p", "src/**/*.py")
        assert "python3 -c" in cmd
        assert "-rHnFZ" not in cmd
        assert "--null" not in cmd

    def test_slash_glob_command_survives_sh_c_wrap(self) -> None:
        # The seatbelt provider wraps every command in `sh -c "cd ... && <cmd>"`.
        # The slash-glob command must survive that wrap: exit 0 and emit a
        # parseable `path\0line:text` record, not a traceback.
        import shlex
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "pkg").mkdir()
            (Path(d) / "pkg" / "deep.py").write_text("needle\n")
            cmd = _build_bsd_grep_cmd("needle", str(Path(d)), "pkg/*.py")
            argv = ["sh", "-c", f"cd {shlex.quote(d)} && {cmd}"]
            p = subprocess.run(argv, capture_output=True, check=False)
            assert p.returncode == 0, p.stderr.decode("utf-8", "replace")
            from deepagents.backends.protocol import ExecuteResponse
            from deepagents.backends.sandbox import _parse_grep_output

            out = (p.stdout + p.stderr).decode("utf-8", errors="replace")
            res = _parse_grep_output(
                ExecuteResponse(output=out, exit_code=p.returncode), str(Path(d)), None
            )
            assert res.error is None, res.error
            assert res.matches is not None, res.matches
            assert len(res.matches) == 1, res.matches
            assert res.matches[0]["path"].endswith("deep.py")
            assert res.matches[0]["text"] == "needle"

    def test_parse_null_output_yields_matches(self) -> None:
        # The base `_parse_grep_output` consumes the `path\0line:text` records
        # that `--null` produces on BSD grep exactly as it does GNU `-Z`.
        from deepagents.backends.protocol import ExecuteResponse
        from deepagents.backends.sandbox import _parse_grep_output

        output = "./b.txt\x002:hello again\n./a.txt\x001:hello world\n"
        result = _parse_grep_output(
            ExecuteResponse(output=output, exit_code=0), ".", None
        )
        assert result.error is None
        assert result.matches is not None
        assert len(result.matches) == 2
        assert result.matches[0]["path"] == "./b.txt"
        assert result.matches[0]["line"] == 2
        assert result.matches[0]["text"] == "hello again"

    def test_plain_colon_output_is_unparseable_by_base_parser(self) -> None:
        # BSD `-Z` (`--decompress`) emits plain `path:line:text` with no NUL,
        # which the base parser cannot split: the failure the override exists
        # to prevent.
        from deepagents.backends.protocol import ExecuteResponse
        from deepagents.backends.sandbox import _parse_grep_output

        bsd_minus_z_output = "./b.txt:2:hello again\n./a.txt:1:hello world\n"
        result = _parse_grep_output(
            ExecuteResponse(output=bsd_minus_z_output, exit_code=0), ".", None
        )
        # With no NUL, the split fails and the last offending line surfaces as
        # an error with empty matches.
        assert result.error is not None
        assert not result.matches


# profile generation (no sandbox-exec needed)


class TestProfileText:
    """The generated SBPL profile reflects the launch-dir model: broad reads,
    /Users deny, launch-dir re-allow + read_paths re-allow, launch-dir-only
    writes, and the configured network rule."""

    def test_profile_contains_broad_reads_and_users_deny(
        self, tmp_path: Path
    ) -> None:
        sb = SeatbeltSandbox("profile-test", tmp_path)
        text = sb._profile_text()
        assert "(allow file-read*)" in text
        assert "(allow file-read-data)" in text
        assert "(allow file-read-metadata)" in text
        assert '(deny file-read*\n    (subpath "/Users"))' in text
        assert '(deny file-read-data\n    (subpath "/Users"))' in text
        assert '(deny file-read-metadata\n    (subpath "/Users"))' in text

    def test_profile_re_allows_launch_dir(self, tmp_path: Path) -> None:
        sb = SeatbeltSandbox("profile-test", tmp_path)
        text = sb._profile_text()
        resolved = str(tmp_path.resolve())
        if resolved.startswith("/Users/"):
            # Under /Users: three re-allow rules (one per read op type).
            assert text.count(f'subpath "{resolved}"') >= 3
        else:
            # Outside /Users: only the write rule references it.
            assert f'(allow file-write*\n    (subpath "{resolved}"))' in text

    def test_profile_writes_confined_to_launch_dir(self, tmp_path: Path) -> None:
        sb = SeatbeltSandbox("profile-test", tmp_path)
        text = sb._profile_text()
        resolved = str(tmp_path.resolve())
        assert f'(allow file-write*\n    (subpath "{resolved}"))' in text
        assert '(literal "/dev/null")' in text
        # Default deny covers /Users writes; no explicit write-deny needed.
        assert '(deny file-write*' not in text

    def test_profile_network_default_allow(self, tmp_path: Path) -> None:
        sb = SeatbeltSandbox("profile-test", tmp_path, network=True)
        assert "(allow network*)" in sb._profile_text()

    def test_profile_network_opt_out(self, tmp_path: Path) -> None:
        sb = SeatbeltSandbox("profile-test", tmp_path, network=False)
        assert "(deny network*)" in sb._profile_text()

    def test_profile_read_paths_re_allowed(self, tmp_path: Path) -> None:
        extra = "/Users/tester/extra"
        sb = SeatbeltSandbox("profile-test", tmp_path, read_paths=[extra])
        text = sb._profile_text()
        assert f'(allow file-read* (subpath "{extra}"))' in text
        assert f'(allow file-read-data (subpath "{extra}"))' in text
        assert f'(allow file-read-metadata (subpath "{extra}"))' in text

    def test_profile_has_ancestor_metadata_when_launch_under_users(
        self, tmp_path: Path
    ) -> None:
        # When the launch dir is under /Users, the profile re-allows metadata
        # on /Users and each ancestor so `cd` can traverse.
        sb = SeatbeltSandbox("profile-test", tmp_path)
        text = sb._profile_text()
        resolved = str(tmp_path.resolve())
        if resolved.startswith("/Users/"):
            assert '(allow file-read-metadata (subpath "/Users"))' in text
            # Each ancestor of the launch dir gets a metadata-only re-allow.
            parts = Path(resolved).parts
            acc = ""
            for part in parts[1:-1]:
                acc += "/" + part
                if acc.startswith("/Users"):
                    assert f'(allow file-read-metadata (subpath "{acc}"))' in text


# environment scrubbing (no sandbox-exec needed)


class TestEnvScrubbing:
    """`_env` returns a minimal, secret-free environment: dcode's own env
    (API keys/tokens) is not inherited."""

    def test_minimal_env_only(self, tmp_path: Path) -> None:
        sb = SeatbeltSandbox("env-test", tmp_path)
        env = sb._env()
        assert set(env) == {"PATH", "HOME", "TMPDIR", "SHELL", "LANG"}

    def test_home_is_real_user_home(self, tmp_path: Path) -> None:
        # HOME is the real user home, not the launch dir: toolchains resolve
        # caches relative to HOME (Go: ~/go; Rust: ~/.cargo; ...). The SBPL
        # profile, not HOME, is the fence that keeps secrets unreadable.
        sb = SeatbeltSandbox("env-test", tmp_path)
        assert sb._env()["HOME"] == str(Path.home())

    def test_tmpdir_inside_launch_dir(self, tmp_path: Path) -> None:
        sb = SeatbeltSandbox("env-test", tmp_path)
        assert sb._env()["TMPDIR"] == str(tmp_path.resolve() / ".tmp")


# provider id validation (no sandbox-exec needed)


class TestProviderIdValidation:
    def test_rejects_separator(self) -> None:
        from dcode_seatbelt_sandbox.provider import SeatbeltProvider

        with pytest.raises(ValueError):
            SeatbeltProvider().get_or_create(sandbox_id="a/b")

    def test_rejects_whitespace(self) -> None:
        from dcode_seatbelt_sandbox.provider import SeatbeltProvider

        with pytest.raises(ValueError):
            SeatbeltProvider().get_or_create(sandbox_id="foo bar")

    def test_delete_rejects_bad_id(self) -> None:
        from dcode_seatbelt_sandbox.provider import SeatbeltProvider

        with pytest.raises(ValueError):
            SeatbeltProvider().delete(sandbox_id="a/b")
