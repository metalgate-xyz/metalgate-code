"""Unit tests for Go contextual symbol search tools in a monorepo layout.

This mirrors the structure of go.evroc.dev: a single module with nested packages
under private/, public/, and e2e-tests/.

These tests use ``LocalShellBackend`` and run on the host without a
sandbox.  Sandbox/agent integration tests live in
``test_go_context_integration.py``.
"""

import shutil
from pathlib import Path

import pytest

from marketplace.plugins.context_tools.extension.cache import CodeCache
from marketplace.plugins.context_tools.extension.factory import _create_tracer
from marketplace.plugins.context_tools.extension.tools import make_tools

# Source-of-truth sample in the repo; the module fixture copies it to a temp
# dir and rebinds these globals to the copy so the tracer's cache (.metalgate/)
# and gopls root land in temp, not in the committed sample.
MONOREPO_SRC = Path(__file__).parent / "sample" / "go" / "monorepo"
MONOREPO_DIR = MONOREPO_SRC
SHARED_FILE = str(
    MONOREPO_SRC / "private" / "service" / "internal" / "shared" / "context.go"
)
RENDERER_FILE = str(
    MONOREPO_SRC / "private" / "service" / "internal" / "shared" / "renderer.go"
)
CONTROLLER_FILE = str(MONOREPO_SRC / "private" / "service" / "api" / "controller.go")
MIDDLEWARE_FILE = str(MONOREPO_SRC / "private" / "service" / "api" / "middleware.go")
RENDER_CALL_FILE = str(MONOREPO_SRC / "private" / "service" / "api" / "render_call.go")
GIN_CALL_FILE = str(MONOREPO_SRC / "private" / "service" / "api" / "gin_call.go")
CLOSURE_FILE = str(MONOREPO_SRC / "private" / "service" / "api" / "closure.go")
CLIENT_FILE = str(MONOREPO_SRC / "public" / "client" / "client.go")
E2E_FILE = str(MONOREPO_SRC / "e2e-tests" / "suite" / "test.go")


@pytest.fixture(scope="module")
def tools(tmp_path_factory):
    global MONOREPO_DIR, SHARED_FILE, RENDERER_FILE, CONTROLLER_FILE
    global MIDDLEWARE_FILE, RENDER_CALL_FILE, GIN_CALL_FILE, CLOSURE_FILE
    global CLIENT_FILE, E2E_FILE

    sample_dir = tmp_path_factory.mktemp("go_mono") / "monorepo"
    shutil.copytree(MONOREPO_SRC, sample_dir)
    MONOREPO_DIR = sample_dir
    SHARED_FILE = str(
        sample_dir / "private" / "service" / "internal" / "shared" / "context.go"
    )
    RENDERER_FILE = str(
        sample_dir / "private" / "service" / "internal" / "shared" / "renderer.go"
    )
    CONTROLLER_FILE = str(sample_dir / "private" / "service" / "api" / "controller.go")
    MIDDLEWARE_FILE = str(sample_dir / "private" / "service" / "api" / "middleware.go")
    RENDER_CALL_FILE = str(
        sample_dir / "private" / "service" / "api" / "render_call.go"
    )
    GIN_CALL_FILE = str(sample_dir / "private" / "service" / "api" / "gin_call.go")
    CLOSURE_FILE = str(sample_dir / "private" / "service" / "api" / "closure.go")
    CLIENT_FILE = str(sample_dir / "public" / "client" / "client.go")
    E2E_FILE = str(sample_dir / "e2e-tests" / "suite" / "test.go")

    cache_path = str(sample_dir / ".metalgate" / "context_cache.db")
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    cache = CodeCache(cache_path)
    tracer = _create_tracer(root=str(sample_dir), cache=cache, language="go")
    (
        goto_def,
        outline,
        get_source,
        callers,
        callees,
        find_sym,
        set_root,
    ) = make_tools(tracer)

    try:
        yield {
            "goto_definition": goto_def,
            "get_file_outline": outline,
            "get_source": get_source,
            "get_callers": callers,
            "get_callees": callees,
            "find_symbol": find_sym,
            "set_language_server_root": set_root,
            "cache": cache,
        }
    finally:
        tracer.stop()


# get_file_outline
class TestGetFileOutline:
    def test_finds_struct(self, tools):
        symbols = tools["get_file_outline"](CONTROLLER_FILE)
        assert any(s["name"] == "Controller" and s["kind"] == "struct" for s in symbols)

    def test_finds_function(self, tools):
        symbols = tools["get_file_outline"](CONTROLLER_FILE)
        assert any(
            s["name"] == "NewController" and s["kind"] == "function" for s in symbols
        )

    def test_finds_method(self, tools):
        symbols = tools["get_file_outline"](CONTROLLER_FILE)
        method = next(
            (s for s in symbols if s["name"] == "Publish" and s["kind"] == "method"),
            None,
        )
        assert method is not None

    def test_method_has_receiver(self, tools):
        symbols = tools["get_file_outline"](CONTROLLER_FILE)
        publish = next(s for s in symbols if s["name"] == "Publish")
        assert "Controller" in (publish.get("class") or "")

    def test_finds_function_in_shared(self, tools):
        symbols = tools["get_file_outline"](SHARED_FILE)
        assert any(
            s["name"] == "ToContext" and s["kind"] == "function" for s in symbols
        )
        assert any(
            s["name"] == "FromContext" and s["kind"] == "function" for s in symbols
        )

    def test_cached_result_is_identical(self, tools):
        first = tools["get_file_outline"](CONTROLLER_FILE)
        second = tools["get_file_outline"](CONTROLLER_FILE)
        assert first == second


# get_source
class TestGetSource:
    def _publish_line(self, tools):
        symbols = tools["get_file_outline"](CONTROLLER_FILE)
        return next(s for s in symbols if s["name"] == "Publish")["line"]

    def test_source_contains_func(self, tools):
        line = self._publish_line(tools)
        result = tools["get_source"](CONTROLLER_FILE, line)
        assert "Publish" in result["source"]
        assert "shared.ToContext" in result["source"]

    def test_start_and_end_lines_are_sane(self, tools):
        line = self._publish_line(tools)
        result = tools["get_source"](CONTROLLER_FILE, line)
        assert result["start_line"] >= 1
        assert result["end_line"] >= result["start_line"]

    def test_get_source_from_body_line(self, tools):
        symbols = tools["get_file_outline"](CONTROLLER_FILE)
        publish = next(s for s in symbols if s["name"] == "Publish")
        body_line = publish["line"] + 2
        result = tools["get_source"](CONTROLLER_FILE, body_line)
        assert "Publish" in result["source"]

    def test_get_source_cross_package(self, tools):
        symbols = tools["get_file_outline"](SHARED_FILE)
        tc = next(s for s in symbols if s["name"] == "ToContext")
        result = tools["get_source"](SHARED_FILE, tc["line"])
        assert "ToContext" in result["source"]


# find_symbol
class TestFindSymbol:
    def test_exact_match_finds_to_context(self, tools):
        results = tools["find_symbol"]("ToContext")
        names = [r["name"] for r in results]
        assert "ToContext" in names

    def test_exact_match_finds_from_context(self, tools):
        results = tools["find_symbol"]("FromContext")
        names = [r["name"] for r in results]
        assert "FromContext" in names

    def test_finds_struct_by_name(self, tools):
        results = tools["find_symbol"]("Controller")
        names = [r["name"] for r in results]
        assert "Controller" in names

    def test_finds_function_cross_package(self, tools):
        results = tools["find_symbol"]("NewController")
        names = [r["name"] for r in results]
        assert "NewController" in names

    def test_unknown_symbol_returns_empty_list(self, tools):
        results = tools["find_symbol"]("zzz_does_not_exist_xyz")
        assert results == []

    def test_cached_result_is_identical(self, tools):
        first = tools["find_symbol"]("ToContext")
        second = tools["find_symbol"]("ToContext")
        assert first == second


# goto_definition — cross-package resolution
#
# These tests verify that goto_definition can resolve symbols defined in
# DIFFERENT packages/directories of the monorepo — not just symbols in the
# same file or stdlib.
#
# Cross-package call sites used in these tests:
#
#     controller.go:17  shared.ToContext  -> context.go:6   (api -> shared)
#     client.go:14      api.NewController -> controller.go:11 (client -> api)
#     client.go:19      c.ctrl.Publish    -> controller.go:16 (client -> api, method)
#     test.go:14        api.NewController -> controller.go:11 (suite -> api)
#     test.go:19        r.ctrl.Publish    -> controller.go:16 (suite -> api, method)
#
# All ground-truth locations were captured from gopls v0.23.0 at the correct
# member column of each call site.
class TestGotoDefinitionCrossPackage:
    """Resolves symbols defined in a different package/directory of the
    monorepo.

    Each call site is in one package directory and the target symbol is
    defined in a different package directory.  If gopls hasn't indexed the
    target package's files, ``goto_definition`` returns an empty ``{}``.
    """

    @staticmethod
    def _basename(path: str) -> str:
        return path.replace("\\", "/").rstrip("/").split("/")[-1]

    # cross-package function: shared.ToContext
    #
    # controller.go (package api) calls shared.ToContext, which is defined
    # in context.go (package shared) in a different directory.

    def test_resolves_cross_package_function(self, tools):
        """shared.ToContext (controller.go:17) must resolve to context.go:6
        in the shared package, not return empty.

        gopls ground truth at the member column (col 16):
            context.go:6  func shared.ToContext(key string, value int) map[string]string
        """
        result = tools["goto_definition"](CONTROLLER_FILE, 17, "shared.ToContext")
        assert result, (
            "got empty result — gopls has not indexed the shared package; "
            "the tool must open all .go files in the package directory before querying"
        )
        assert self._basename(result["file"]) == "context.go", (
            f"expected context.go, got {result.get('file', 'EMPTY')}"
        )
        assert result["line"] == 6, f"expected line 6, got {result.get('line')}"
        assert "ToContext" in result["signature"], (
            f"expected signature containing 'ToContext', "
            f"got {result.get('signature', '')!r}"
        )

    def test_cross_package_function_kind(self, tools):
        """shared.ToContext must be classified as a function."""
        result = tools["goto_definition"](CONTROLLER_FILE, 17, "shared.ToContext")
        assert result, "got empty result"
        assert result["kind"] == "function", (
            f"expected kind 'function', got {result['kind']!r}"
        )

    def test_cross_package_function_not_import_line(self, tools):
        """shared.ToContext must NOT resolve to the import line in
        controller.go (line 4, signature 'package shared')."""
        result = tools["goto_definition"](CONTROLLER_FILE, 17, "shared.ToContext")
        assert result, "got empty result"
        assert not (
            self._basename(result["file"]) == "controller.go" and result["line"] == 4
        ), (
            "resolved to the import line — column points at the qualifier, "
            "not the member"
        )

    # cross-package function: api.NewController
    #
    # client.go (package client) calls api.NewController, which is defined
    # in controller.go (package api) in a different directory.

    def test_resolves_cross_package_function_from_client(self, tools):
        """api.NewController (client.go:14) must resolve to controller.go:11
        in the api package, not return empty.

        gopls ground truth at the member column (col 27):
            controller.go:11  func api.NewController() *api.Controller
        """
        result = tools["goto_definition"](CLIENT_FILE, 14, "api.NewController")
        assert result, (
            "got empty result — gopls has not indexed the api package; "
            "the tool must open all .go files in the package directory before querying"
        )
        assert self._basename(result["file"]) == "controller.go", (
            f"expected controller.go, got {result.get('file', 'EMPTY')}"
        )
        assert result["line"] == 11, f"expected line 11, got {result.get('line')}"
        assert "NewController" in result["signature"], (
            f"expected signature containing 'NewController', "
            f"got {result.get('signature', '')!r}"
        )

    # cross-package method: c.ctrl.Publish
    #
    # client.go (package client) calls c.ctrl.Publish, where Publish is a
    # method on *Controller defined in controller.go (package api).
    # This is a chained selector: c.ctrl is a field, .Publish is the method.

    def test_resolves_cross_package_method_from_client(self, tools):
        """c.ctrl.Publish (client.go:19) must resolve to controller.go:16
        (the Publish method), not return empty and not resolve to the
        field declaration.

        gopls ground truth at the member column (col 16, the 'P' of Publish):
            controller.go:16  func (c *api.Controller) Publish(key string, val int) ...
        """
        result = tools["goto_definition"](CLIENT_FILE, 19, "c.ctrl.Publish")
        assert result, (
            "got empty result — gopls has not indexed the api package; "
            "the tool must open all .go files in the package directory before querying"
        )
        assert self._basename(result["file"]) == "controller.go", (
            f"expected controller.go, got {result.get('file', 'EMPTY')}"
        )
        assert result["line"] == 16, f"expected line 16, got {result.get('line')}"
        assert "Publish" in result["signature"], (
            f"expected signature containing 'Publish', "
            f"got {result.get('signature', '')!r}"
        )

    def test_cross_package_method_kind(self, tools):
        """c.ctrl.Publish must be classified as a method."""
        result = tools["goto_definition"](CLIENT_FILE, 19, "c.ctrl.Publish")
        assert result, "got empty result"
        assert result["kind"] == "method", (
            f"expected kind 'method', got {result['kind']!r}"
        )

    def test_cross_package_method_not_field_declaration(self, tools):
        """c.ctrl.Publish must NOT resolve to the field declaration
        'field ctrl *api.Controller' in client.go:9."""
        result = tools["goto_definition"](CLIENT_FILE, 19, "c.ctrl.Publish")
        assert result, "got empty result"
        assert "field " not in result["signature"], (
            f"resolved to a field declaration: {result['signature']!r} "
            "(column points at the receiver/field, not the method)"
        )

    # cross-package from e2e-tests
    #
    # test.go (package suite) calls api.NewController and r.ctrl.Publish,
    # both defined in controller.go (package api).  The e2e-tests directory
    # is a separate package that imports the api package.

    def test_resolves_cross_package_function_from_e2e(self, tools):
        """api.NewController (test.go:14) must resolve to controller.go:11
        from the e2e-tests package, not return empty.

        gopls ground truth at the member column (col 27):
            controller.go:11  func api.NewController() *api.Controller
        """
        result = tools["goto_definition"](E2E_FILE, 14, "api.NewController")
        assert result, (
            "got empty result — gopls has not indexed the api package; "
            "the tool must open all .go files in the package directory before querying"
        )
        assert self._basename(result["file"]) == "controller.go", (
            f"expected controller.go, got {result.get('file', 'EMPTY')}"
        )
        assert result["line"] == 11, f"expected line 11, got {result.get('line')}"

    def test_resolves_cross_package_method_from_e2e(self, tools):
        """r.ctrl.Publish (test.go:19) must resolve to controller.go:16
        from the e2e-tests package, not return empty.

        gopls ground truth at the member column (col 16, the 'P' of Publish):
            controller.go:16  func (c *api.Controller) Publish(key string, val int) ...
        """
        result = tools["goto_definition"](E2E_FILE, 19, "r.ctrl.Publish")
        assert result, (
            "got empty result — gopls has not indexed the api package; "
            "the tool must open all .go files in the package directory before querying"
        )
        assert self._basename(result["file"]) == "controller.go", (
            f"expected controller.go, got {result.get('file', 'EMPTY')}"
        )
        assert result["line"] == 16, f"expected line 16, got {result.get('line')}"
        assert "Publish" in result["signature"], (
            f"expected signature containing 'Publish', "
            f"got {result.get('signature', '')!r}"
        )


# goto_definition — same-package, different file
#
# This reproduces the gin bug where a method call like `c.Next` in
# recovery.go (package gin) must resolve to context.go (also package gin,
# but a different file).  The original cross-package tests only exercised
# calls that cross package boundaries (different import paths).  The
# same-package-different-file case is a distinct failure mode: gopls
# must have the sibling file indexed to resolve the target.
#
# middleware.go (package api) calls c.Publish and c.Lookup, both defined
# in controller.go (also package api, different file).
#
# Ground truth (controller.go):
#     line 16: func (c *Controller) Publish(key string, val int) map[string]string
#     line 21: func (c *Controller) Lookup(ctx map[string]string, key string) (string, error)
class TestGotoDefinitionSamePackage:
    """Resolves method calls within the same package but a different file.

    middleware.go (package api) calls c.Publish and c.Lookup, both defined
    in controller.go (also package api).  If gopls hasn't indexed
    controller.go, ``goto_definition`` returns an empty ``{}`` or resolves
    to the parameter declaration instead of the method.
    """

    @staticmethod
    def _basename(path: str) -> str:
        return path.replace("\\", "/").rstrip("/").split("/")[-1]

    # c.Publish: middleware.go:8 -> controller.go:16

    def test_resolves_same_package_method_publish(self, tools):
        """c.Publish (middleware.go:8) must resolve to controller.go:16
        (the Publish method in the same package, different file), not
        return empty and not resolve to the parameter declaration.

        gopls ground truth at the member column (col 11, the 'P' of Publish):
            controller.go:16  func (c *Controller) Publish(key string, val int) ...
        """
        result = tools["goto_definition"](MIDDLEWARE_FILE, 8, "c.Publish")
        assert result, (
            "got empty result — gopls has not indexed controller.go; "
            "the tool must open all .go files in the package directory before querying"
        )
        assert self._basename(result["file"]) == "controller.go", (
            f"expected controller.go, got {result.get('file', 'EMPTY')}"
        )
        assert result["line"] == 16, f"expected line 16, got {result.get('line')}"

    def test_same_package_method_publish_signature(self, tools):
        """c.Publish must return a signature containing 'Publish'."""
        result = tools["goto_definition"](MIDDLEWARE_FILE, 8, "c.Publish")
        assert result, "got empty result"
        assert "Publish" in result["signature"], (
            f"expected signature containing 'Publish', "
            f"got {result.get('signature', '')!r}"
        )

    def test_same_package_method_publish_kind(self, tools):
        """c.Publish must be classified as a method."""
        result = tools["goto_definition"](MIDDLEWARE_FILE, 8, "c.Publish")
        assert result, "got empty result"
        assert result["kind"] == "method", (
            f"expected kind 'method', got {result['kind']!r}"
        )

    def test_same_package_method_publish_not_param(self, tools):
        """c.Publish must NOT resolve to the parameter declaration in
        middleware.go (line 7, 'c *Controller')."""
        result = tools["goto_definition"](MIDDLEWARE_FILE, 8, "c.Publish")
        assert result, "got empty result"
        assert not (
            self._basename(result["file"]) == "middleware.go" and result["line"] == 7
        ), (
            "resolved to the parameter declaration — column points at the "
            "receiver, not the member"
        )
        assert "var " not in result.get("signature", ""), (
            f"resolved to a variable declaration: {result['signature']!r}"
        )

    # c.Lookup: middleware.go:14 -> controller.go:21

    def test_resolves_same_package_method_lookup(self, tools):
        """c.Lookup (middleware.go:14) must resolve to controller.go:21
        (the Lookup method in the same package, different file).

        gopls ground truth at the member column (col 11, the 'L' of Lookup):
            controller.go:21  func (c *Controller) Lookup(ctx map[string]string, ...) ...
        """
        result = tools["goto_definition"](MIDDLEWARE_FILE, 14, "c.Lookup")
        assert result, (
            "got empty result — gopls has not indexed controller.go; "
            "the tool must open all .go files in the package directory before querying"
        )
        assert self._basename(result["file"]) == "controller.go", (
            f"expected controller.go, got {result.get('file', 'EMPTY')}"
        )
        assert result["line"] == 21, f"expected line 21, got {result.get('line')}"

    def test_same_package_method_lookup_signature(self, tools):
        """c.Lookup must return a signature containing 'Lookup'."""
        result = tools["goto_definition"](MIDDLEWARE_FILE, 14, "c.Lookup")
        assert result, "got empty result"
        assert "Lookup" in result["signature"], (
            f"expected signature containing 'Lookup', "
            f"got {result.get('signature', '')!r}"
        )

    def test_same_package_method_lookup_kind(self, tools):
        """c.Lookup must be classified as a method."""
        result = tools["goto_definition"](MIDDLEWARE_FILE, 14, "c.Lookup")
        assert result, "got empty result"
        assert result["kind"] == "method", (
            f"expected kind 'method', got {result['kind']!r}"
        )

    def test_same_package_method_col_points_at_member(self, tools):
        """For c.Publish on middleware.go:8, the returned col must point
        at the 'Publish' member (col 11), not the receiver 'c' or the dot."""
        result = tools["goto_definition"](MIDDLEWARE_FILE, 8, "c.Publish")
        assert result, "got empty result"
        assert 11 <= result["col"] <= 18, (
            f"expected col in [11, 18] (the 'Publish' member), got col={result['col']}"
        )


# goto_definition — interface method called across packages
#
# This reproduces the gin bug where `r.Render` in context.go (package gin)
# must resolve to the interface method in render.go (package render).
# gopls resolves interface-method calls to the method line INSIDE the
# interface, not the `type X interface` line.
#
# render_call.go (package api) calls r.Render and r.WriteContentType on
# a shared.Renderer interface, defined in renderer.go (package shared).
#
# Ground truth (renderer.go):
#     line 10: 	Render(dest string) error          (interface method)
#     line 12: 	WriteContentType() string          (interface method)
class TestGotoDefinitionInterfaceMethod:
    """Resolves interface method calls across packages.

    render_call.go (package api) calls r.Render and r.WriteContentType
    on a variable of type shared.Renderer (an interface).  The methods
    are defined inside the interface in renderer.go (package shared).

    gopls resolves to the method line inside the interface, NOT the
    `type Renderer interface` line.  The hover signature starts with
    `func (shared.Renderer)` so kind inference must classify it as a method.
    """

    @staticmethod
    def _basename(path: str) -> str:
        return path.replace("\\", "/").rstrip("/").split("/")[-1]

    # r.WriteContentType: render_call.go:12 -> renderer.go:12

    def test_resolves_interface_method_write_content_type(self, tools):
        """r.WriteContentType (render_call.go:12) must resolve to
        renderer.go:12 (the method line inside the interface), not the
        `type Renderer interface` line and not the parameter declaration.

        gopls ground truth at the member column (col 10, the 'W'):
            renderer.go:12  WriteContentType() string
        """
        result = tools["goto_definition"](RENDER_CALL_FILE, 12, "r.WriteContentType")
        assert result, (
            "got empty result — gopls has not indexed the shared package; "
            "the tool must open all .go files in the package directory before querying"
        )
        assert self._basename(result["file"]) == "renderer.go", (
            f"expected renderer.go, got {result.get('file', 'EMPTY')}"
        )
        assert result["line"] == 12, f"expected line 12, got {result.get('line')}"

    def test_interface_method_write_content_type_signature(self, tools):
        """r.WriteContentType must return a signature containing
        'WriteContentType'."""
        result = tools["goto_definition"](RENDER_CALL_FILE, 12, "r.WriteContentType")
        assert result, "got empty result"
        assert "WriteContentType" in result["signature"], (
            f"expected signature containing 'WriteContentType', "
            f"got {result.get('signature', '')!r}"
        )

    def test_interface_method_write_content_type_kind(self, tools):
        """r.WriteContentType must be classified as a method (the hover
        signature starts with 'func (shared.Renderer)')."""
        result = tools["goto_definition"](RENDER_CALL_FILE, 12, "r.WriteContentType")
        assert result, "got empty result"
        assert result["kind"] == "method", (
            f"expected kind 'method', got {result['kind']!r}"
        )

    def test_interface_method_write_content_type_not_type_line(self, tools):
        """r.WriteContentType must NOT resolve to the `type Renderer
        interface` line (renderer.go:8).  gopls resolves to the method
        line inside the interface, not the type declaration."""
        result = tools["goto_definition"](RENDER_CALL_FILE, 12, "r.WriteContentType")
        assert result, "got empty result"
        assert not (
            self._basename(result["file"]) == "renderer.go" and result["line"] == 8
        ), (
            "resolved to the 'type Renderer interface' line — should resolve "
            "to the method line inside the interface"
        )

    def test_interface_method_write_content_type_not_param(self, tools):
        """r.WriteContentType must NOT resolve to the parameter declaration
        in render_call.go (line 11, 'r shared.Renderer')."""
        result = tools["goto_definition"](RENDER_CALL_FILE, 12, "r.WriteContentType")
        assert result, "got empty result"
        assert not (
            self._basename(result["file"]) == "render_call.go" and result["line"] == 11
        ), (
            "resolved to the parameter declaration — column points at the "
            "receiver, not the member"
        )

    # r.Render: render_call.go:14 -> renderer.go:10

    def test_resolves_interface_method_render(self, tools):
        """r.Render (render_call.go:14) must resolve to renderer.go:10
        (the method line inside the interface), not the `type Renderer
        interface` line and not the parameter declaration.

        gopls ground truth at the member column (col 11, the 'R'):
            renderer.go:10  Render(dest string) error
        """
        result = tools["goto_definition"](RENDER_CALL_FILE, 14, "r.Render")
        assert result, (
            "got empty result — gopls has not indexed the shared package; "
            "the tool must open all .go files in the package directory before querying"
        )
        assert self._basename(result["file"]) == "renderer.go", (
            f"expected renderer.go, got {result.get('file', 'EMPTY')}"
        )
        assert result["line"] == 10, f"expected line 10, got {result.get('line')}"

    def test_interface_method_render_signature(self, tools):
        """r.Render must return a signature containing 'Render'."""
        result = tools["goto_definition"](RENDER_CALL_FILE, 14, "r.Render")
        assert result, "got empty result"
        assert "Render" in result["signature"], (
            f"expected signature containing 'Render', "
            f"got {result.get('signature', '')!r}"
        )

    def test_interface_method_render_kind(self, tools):
        """r.Render must be classified as a method."""
        result = tools["goto_definition"](RENDER_CALL_FILE, 14, "r.Render")
        assert result, "got empty result"
        assert result["kind"] == "method", (
            f"expected kind 'method', got {result['kind']!r}"
        )

    def test_interface_method_render_not_type_line(self, tools):
        """r.Render must NOT resolve to the `type Renderer interface` line
        (renderer.go:8)."""
        result = tools["goto_definition"](RENDER_CALL_FILE, 14, "r.Render")
        assert result, "got empty result"
        assert not (
            self._basename(result["file"]) == "renderer.go" and result["line"] == 8
        ), (
            "resolved to the 'type Renderer interface' line — should resolve "
            "to the method line inside the interface"
        )

    def test_interface_method_col_points_at_member(self, tools):
        """For r.Render on render_call.go:14, the returned col must point
        at the 'Render' member (col 11), not the receiver 'r' or the dot."""
        result = tools["goto_definition"](RENDER_CALL_FILE, 14, "r.Render")
        assert result, "got empty result"
        assert 11 <= result["col"] <= 17, (
            f"expected col in [11, 17] (the 'Render' member), got col={result['col']}"
        )


# goto_definition — 3rd-party package qualified call
#
# This reproduces the gin bug where `gin.New()` in external.go must resolve
# to the function definition in the 3rd-party gin package, not the
# `import "github.com/gin-gonic/gin"` line in the caller.
#
# gin_call.go (package api) imports gin and calls gin.New() and gin.Default().
#
# Ground truth (gin v1.12.0):
#     gin.go:202  func New(opts ...OptionFunc) *Engine
#     gin.go:236  func Default(opts ...OptionFunc) *Engine
class TestGotoDefinitionThirdParty:
    """Resolves qualified calls to 3rd-party package functions.

    gin_call.go (package api) imports gin and calls gin.New() and
    gin.Default().  The tool must resolve to the function definition in
    the gin package source, not the `import "github.com/gin-gonic/gin"`
    line in gin_call.go.
    """

    @staticmethod
    def _basename(path: str) -> str:
        return path.replace("\\", "/").rstrip("/").split("/")[-1]

    # gin.New: gin_call.go:11 -> gin.go:202

    def test_resolves_third_party_gin_new(self, tools):
        """gin.New (gin_call.go:11) must resolve to gin.go (the 3rd-party
        package source), not gin_call.go:4 (the import line in the caller).

        gopls ground truth at the member column (col 13, the 'N'):
            gin.go:202  func New(opts ...OptionFunc) *Engine
        """
        result = tools["goto_definition"](GIN_CALL_FILE, 11, "gin.New")
        assert result, "got empty result"
        assert self._basename(result["file"]) == "gin.go", (
            f"expected gin.go (the function definition), "
            f"got {result.get('file', 'EMPTY')} "
            "(resolving the package import instead of the function)"
        )
        assert result["line"] == 202, f"expected line 202, got {result.get('line')}"

    def test_third_party_gin_new_signature(self, tools):
        """gin.New must return a signature containing 'New'."""
        result = tools["goto_definition"](GIN_CALL_FILE, 11, "gin.New")
        assert result, "got empty result"
        assert "New" in result["signature"], (
            f"expected signature containing 'New', got {result.get('signature', '')!r}"
        )

    def test_third_party_gin_new_kind(self, tools):
        """gin.New must be classified as a function, not 'unknown'."""
        result = tools["goto_definition"](GIN_CALL_FILE, 11, "gin.New")
        assert result, "got empty result"
        assert result["kind"] == "function", (
            f"expected kind 'function', got {result['kind']!r} "
            "(kind inference fell back to 'unknown' because the hover "
            "signature is the package docstring, not the function signature)"
        )

    def test_third_party_gin_new_not_import_line(self, tools):
        """gin.New must NOT resolve to the import line in gin_call.go
        (line 4, 'github.com/gin-gonic/gin')."""
        result = tools["goto_definition"](GIN_CALL_FILE, 11, "gin.New")
        assert result, "got empty result"
        assert not (
            self._basename(result["file"]) == "gin_call.go" and result["line"] == 4
        ), (
            "resolved to the import line — the column points at the "
            "qualifier/dot, not the member 'New'"
        )

    def test_third_party_gin_new_col_points_at_member(self, tools):
        """For gin.New on gin_call.go:11, the returned col must point at
        the 'New' member (col 13), not the qualifier 'gin' or the dot."""
        result = tools["goto_definition"](GIN_CALL_FILE, 11, "gin.New")
        assert result, "got empty result"
        assert 13 <= result["col"] <= 16, (
            f"expected col in [13, 16] (the 'New' member), got col={result['col']}"
        )

    # gin.Default: gin_call.go:16 -> gin.go:236

    def test_resolves_third_party_gin_default(self, tools):
        """gin.Default (gin_call.go:16) must resolve to gin.go:236.

        gopls ground truth at the member column (col 13, the 'D'):
            gin.go:236  func Default(opts ...OptionFunc) *Engine
        """
        result = tools["goto_definition"](GIN_CALL_FILE, 16, "gin.Default")
        assert result, "got empty result"
        assert self._basename(result["file"]) == "gin.go", (
            f"expected gin.go, got {result.get('file', 'EMPTY')}"
        )
        assert result["line"] == 236, f"expected line 236, got {result.get('line')}"

    def test_third_party_gin_default_signature(self, tools):
        """gin.Default must return a signature containing 'Default'."""
        result = tools["goto_definition"](GIN_CALL_FILE, 16, "gin.Default")
        assert result, "got empty result"
        assert "Default" in result["signature"], (
            f"expected signature containing 'Default', "
            f"got {result.get('signature', '')!r}"
        )

    def test_third_party_gin_default_kind(self, tools):
        """gin.Default must be classified as a function."""
        result = tools["goto_definition"](GIN_CALL_FILE, 16, "gin.Default")
        assert result, "got empty result"
        assert result["kind"] == "function", (
            f"expected kind 'function', got {result['kind']!r}"
        )

    def test_third_party_gin_default_not_import_line(self, tools):
        """gin.Default must NOT resolve to the import line in gin_call.go."""
        result = tools["goto_definition"](GIN_CALL_FILE, 16, "gin.Default")
        assert result, "got empty result"
        assert not (
            self._basename(result["file"]) == "gin_call.go" and result["line"] == 4
        ), (
            "resolved to the import line — the column points at the "
            "qualifier/dot, not the member 'Default'"
        )


# goto_definition — method calls inside function literals (closures)
#
# This reproduces the core gin bug.  In gin's recovery.go,
# CustomRecoveryWithWriter returns a func(c *Context) (a HandlerFunc).
# Inside that returned closure, calls like c.Next(), c.Error(), c.Abort()
# all return empty {} from goto_definition, while the same calls in a
# non-closure function resolve correctly.
#
# The bug: tree-sitter fails to locate the selector_expression node when
# it is inside a function literal body that is returned, assigned, or
# passed as an argument.  The column computation returns a position that
# doesn't correspond to any identifier, so gopls returns empty {}.
#
# closure.go (package api) has five patterns:
#
#     line 21:  return c.Publish("returned", 1)    inside a RETURNED closure
#     line 29:  return c.Publish("assigned", 1)    inside an ASSIGNED closure
#     line 38:  return c.Publish("passed", 1)       inside a PASSED closure
#     line 51:  _ = c.Publish("immediate", 1)       inside an immediately-invoked closure
#     line 58:  return c.Publish("direct", 1)       direct call, no closure (control)
#
# Ground truth: all five must resolve to controller.go:16
#     func (c *Controller) Publish(key string, val int) map[string]string
#
# The first three (returned/assigned/passed) reproduce the bug.
# The last two (immediate-invoke, direct) are control cases that already work.
class TestGotoDefinitionInClosure:
    """Resolves method calls inside function literals (closures).

    closure.go has five call sites for c.Publish, all targeting
    controller.go:16.  The first three are inside function literals
    that are returned, assigned, or passed as an argument.  The last
    two are control cases (immediately-invoked closure and direct call).

    The bug: goto_definition returns empty {} for calls inside
    non-immediately-invoked function literals because tree-sitter
    fails to locate the selector_expression node, causing the column
    computation to return a position that doesn't correspond to any
    identifier.
    """

    @staticmethod
    def _basename(path: str) -> str:
        return path.replace("\\", "/").rstrip("/").split("/")[-1]

    # returned closure: closure.go:21

    def test_resolves_method_in_returned_closure(self, tools):
        """c.Publish inside a returned func literal (closure.go:21) must
        resolve to controller.go:16, not return empty.

        This reproduces gin's recovery.go pattern where
        CustomRecoveryWithWriter returns a func(c *Context) and calls
        like c.Next() inside that closure return {}.
        """
        result = tools["goto_definition"](CLOSURE_FILE, 21, "c.Publish")
        assert result, (
            "got empty result — tree-sitter failed to locate the "
            "selector_expression node inside a returned function literal; "
            "the column computation returned a non-identifier position"
        )
        assert self._basename(result["file"]) == "controller.go", (
            f"expected controller.go, got {result.get('file', 'EMPTY')}"
        )
        assert result["line"] == 16, f"expected line 16, got {result.get('line')}"
        assert "Publish" in result["signature"], (
            f"expected signature containing 'Publish', "
            f"got {result.get('signature', '')!r}"
        )

    def test_returned_closure_kind_is_method(self, tools):
        """c.Publish inside a returned closure must be classified as a method."""
        result = tools["goto_definition"](CLOSURE_FILE, 21, "c.Publish")
        assert result, "got empty result"
        assert result["kind"] == "method", (
            f"expected kind 'method', got {result['kind']!r}"
        )

    def test_returned_closure_col_points_at_member(self, tools):
        """For c.Publish inside a returned closure, the returned col must
        point at the 'Publish' member, not the receiver or dot."""
        result = tools["goto_definition"](CLOSURE_FILE, 21, "c.Publish")
        assert result, "got empty result"
        assert 12 <= result["col"] <= 19, (
            f"expected col in [12, 19] (the 'Publish' member), got col={result['col']}"
        )

    # assigned closure: closure.go:29

    def test_resolves_method_in_assigned_closure(self, tools):
        """c.Publish inside an assigned func literal (closure.go:29) must
        resolve to controller.go:16, not return empty."""
        result = tools["goto_definition"](CLOSURE_FILE, 29, "c.Publish")
        assert result, (
            "got empty result — tree-sitter failed to locate the "
            "selector_expression node inside an assigned function literal"
        )
        assert self._basename(result["file"]) == "controller.go", (
            f"expected controller.go, got {result.get('file', 'EMPTY')}"
        )
        assert result["line"] == 16, f"expected line 16, got {result.get('line')}"
        assert "Publish" in result["signature"], (
            f"expected signature containing 'Publish', "
            f"got {result.get('signature', '')!r}"
        )

    def test_assigned_closure_kind_is_method(self, tools):
        """c.Publish inside an assigned closure must be classified as a method."""
        result = tools["goto_definition"](CLOSURE_FILE, 29, "c.Publish")
        assert result, "got empty result"
        assert result["kind"] == "method", (
            f"expected kind 'method', got {result['kind']!r}"
        )

    # passed closure: closure.go:38

    def test_resolves_method_in_passed_closure(self, tools):
        """c.Publish inside a func literal passed as an argument
        (closure.go:38) must resolve to controller.go:16, not return empty."""
        result = tools["goto_definition"](CLOSURE_FILE, 38, "c.Publish")
        assert result, (
            "got empty result — tree-sitter failed to locate the "
            "selector_expression node inside a passed function literal"
        )
        assert self._basename(result["file"]) == "controller.go", (
            f"expected controller.go, got {result.get('file', 'EMPTY')}"
        )
        assert result["line"] == 16, f"expected line 16, got {result.get('line')}"
        assert "Publish" in result["signature"], (
            f"expected signature containing 'Publish', "
            f"got {result.get('signature', '')!r}"
        )

    def test_passed_closure_kind_is_method(self, tools):
        """c.Publish inside a passed closure must be classified as a method."""
        result = tools["goto_definition"](CLOSURE_FILE, 38, "c.Publish")
        assert result, "got empty result"
        assert result["kind"] == "method", (
            f"expected kind 'method', got {result['kind']!r}"
        )

    # control: immediately-invoked closure: closure.go:51

    def test_resolves_method_in_immediate_invoke_closure(self, tools):
        """c.Publish inside an immediately-invoked func literal
        (closure.go:51) must resolve to controller.go:16.  This is a
        control case that already works."""
        result = tools["goto_definition"](CLOSURE_FILE, 51, "c.Publish")
        assert result, "got empty result"
        assert self._basename(result["file"]) == "controller.go", (
            f"expected controller.go, got {result.get('file', 'EMPTY')}"
        )
        assert result["line"] == 16, f"expected line 16, got {result.get('line')}"

    # control: direct call: closure.go:58

    def test_resolves_method_direct_call(self, tools):
        """c.Publish as a direct call with no closure (closure.go:58) must
        resolve to controller.go:16.  This is a control case that already
        works."""
        result = tools["goto_definition"](CLOSURE_FILE, 58, "c.Publish")
        assert result, "got empty result"
        assert self._basename(result["file"]) == "controller.go", (
            f"expected controller.go, got {result.get('file', 'EMPTY')}"
        )
        assert result["line"] == 16, f"expected line 16, got {result.get('line')}"

    # consistency: all five patterns resolve to the same target

    def test_all_closure_patterns_resolve_same_target(self, tools):
        """All five c.Publish call sites in closure.go must resolve to the
        same target (controller.go:16).  If any returns empty or a different
        target, the column computation is inconsistent for function literals."""
        for line, label in [
            (21, "returned closure"),
            (29, "assigned closure"),
            (38, "passed closure"),
            (51, "immediate-invoke closure"),
            (58, "direct call"),
        ]:
            result = tools["goto_definition"](CLOSURE_FILE, line, "c.Publish")
            assert result, (
                f"got empty result for {label} at line {line} — "
                f"tree-sitter failed to locate the selector_expression"
            )
            assert self._basename(result["file"]) == "controller.go", (
                f"{label} at line {line}: expected controller.go, "
                f"got {result.get('file', 'EMPTY')}"
            )
            assert result["line"] == 16, (
                f"{label} at line {line}: expected line 16, got {result.get('line')}"
            )


# get_callees — cross-package resolution
#
# Publish (controller.go:16, package api) calls shared.ToContext, defined in
# context.go (package shared) in a different directory.  This is the
# cross-package callee case: get_callees must resolve the call to the shared
# package's definition, not leave it empty or resolve to the import line.
#
# gopls ground truth at the member column of the call site (controller.go:17):
#     context.go:6  func shared.ToContext(key string, value int) map[string]string
class TestGetCalleesCrossPackage:
    """Lists callees that resolve across package/directory boundaries.

    Publish (controller.go:16) calls shared.ToContext in a different
    package.  The callee must resolve to context.go:6 in the shared
    package.
    """

    @staticmethod
    def _basename(path: str) -> str:
        return path.replace("\\", "/").rstrip("/").split("/")[-1]

    def _publish_line(self, tools):
        symbols = tools["get_file_outline"](CONTROLLER_FILE)
        return next(s for s in symbols if s["name"] == "Publish")["line"]

    def test_resolves_cross_package_callee(self, tools):
        """shared.ToContext must appear among Publish's callees, resolved to
        context.go:6, not empty.

        gopls ground truth at the member column (controller.go:17, col 16):
            context.go:6  func shared.ToContext(key string, value int) ...
        """
        line = self._publish_line(tools)
        callees = tools["get_callees"](CONTROLLER_FILE, line)
        tc = next((c for c in callees if c["name"] == "ToContext"), None)
        assert tc is not None, (
            f"ToContext not in callees — got {[c['name'] for c in callees]!r}; "
            "gopls has not indexed the shared package"
        )
        assert self._basename(tc["file"]) == "context.go", (
            f"expected context.go, got {tc['file']}"
        )
        assert tc["line"] == 6, f"expected line 6, got {tc['line']}"

    def test_cross_package_callee_kind_is_function(self, tools):
        """ToContext must be classified as a function."""
        line = self._publish_line(tools)
        callees = tools["get_callees"](CONTROLLER_FILE, line)
        tc = next((c for c in callees if c["name"] == "ToContext"), None)
        assert tc is not None, "ToContext not in callees"
        assert tc["kind"] == "function", f"expected kind 'function', got {tc['kind']!r}"

    def test_callees_have_required_keys(self, tools):
        """Every callee dict must have name, kind, file, line, signature."""
        line = self._publish_line(tools)
        callees = tools["get_callees"](CONTROLLER_FILE, line)
        for c in callees:
            for field in ("name", "kind", "file", "line", "signature"):
                assert field in c, f"missing field {field!r} in callee {c!r}"
            assert c["line"] >= 1, f"callee line must be >= 1, got {c['line']}"


# get_callees — same-package, different file
#
# Middleware (middleware.go:8, package api) calls c.Publish, defined in
# controller.go (also package api, different file).  This mirrors gin's
# recovery.go calling c.Next defined in context.go — same package, sibling
# file.  The callee must resolve across the file boundary within the
# package.
#
# gopls ground truth at the member column (middleware.go:8, col 11):
#     controller.go:16  func (c *Controller) Publish(key string, val int) ...
class TestGetCalleesSamePackage:
    """Lists callees within the same package but a different file.

    Middleware (middleware.go:8) calls c.Publish defined in controller.go.
    The callee must resolve to controller.go:16.
    """

    @staticmethod
    def _basename(path: str) -> str:
        return path.replace("\\", "/").rstrip("/").split("/")[-1]

    def test_resolves_same_package_callee(self, tools):
        """c.Publish must resolve to controller.go:16, not empty."""
        symbols = tools["get_file_outline"](MIDDLEWARE_FILE)
        mw = next(s for s in symbols if s["name"] == "Middleware")
        callees = tools["get_callees"](MIDDLEWARE_FILE, mw["line"])
        pub = next((c for c in callees if c["name"] == "Publish"), None)
        assert pub is not None, (
            f"Publish not in callees — got {[c['name'] for c in callees]!r}"
        )
        assert self._basename(pub["file"]) == "controller.go", (
            f"expected controller.go, got {pub['file']}"
        )
        assert pub["line"] == 16, f"expected line 16, got {pub['line']}"
        assert pub["kind"] == "method", f"expected kind 'method', got {pub['kind']!r}"


# get_callers — cross-package + same-package + e2e
#
# Publish (controller.go:16) is the cross-package hub.  It is called from:
#     middleware.go:8      Middleware(...)              (same package, sibling file)
#     client.go:19         c.ctrl.Publish(...)          (client package)
#     test.go:19           r.ctrl.Publish(...)           (suite package, e2e-tests)
#
# get_callers uses LSP call hierarchy.  Each result must point at the call
# site, resolve the enclosing caller name, and exclude the definition line.
# The e2e-tests case is the one gin exercises: a separate test package
# calling into the api package.
class TestGetCallers:
    """Finds every site that calls the symbol defined on `line` of `file`,
    across package and directory boundaries.

    Publish (controller.go:16) is called from Middleware, Do, and RunTest
    across three packages (api, client, suite).
    """

    @staticmethod
    def _basename(path: str) -> str:
        return path.replace("\\", "/").rstrip("/").split("/")[-1]

    def _publish_line(self, tools):
        symbols = tools["get_file_outline"](CONTROLLER_FILE)
        return next(s for s in symbols if s["name"] == "Publish")["line"]

    def test_finds_all_three_callers(self, tools):
        """Publish must have at least three callers: Middleware, Do, RunTest."""
        line = self._publish_line(tools)
        callers = tools["get_callers"](CONTROLLER_FILE, line)
        callers_by_name = {c.get("caller", "") for c in callers}
        for expected in ("Middleware", "Do", "RunTest"):
            assert expected in callers_by_name, (
                f"expected caller {expected!r}, got {callers_by_name!r}"
            )

    def test_middleware_caller_site(self, tools):
        """Middleware (middleware.go:8) calls c.Publish.  The caller result
        must point at middleware.go line 8."""
        line = self._publish_line(tools)
        callers = tools["get_callers"](CONTROLLER_FILE, line)
        mw = next((c for c in callers if c.get("caller") == "Middleware"), None)
        assert mw is not None, "Middleware not among callers"
        assert self._basename(mw["file"]) == "middleware.go", (
            f"expected middleware.go, got {mw['file']}"
        )
        assert mw["line"] == 8, f"expected line 8, got {mw['line']}"

    def test_client_caller_site(self, tools):
        """Do (client.go:19, package client) calls c.ctrl.Publish.  The
        caller result must point at client.go line 19 — cross-package."""
        line = self._publish_line(tools)
        callers = tools["get_callers"](CONTROLLER_FILE, line)
        do = next((c for c in callers if c.get("caller") == "Do"), None)
        assert do is not None, "Do not among callers"
        assert self._basename(do["file"]) == "client.go", (
            f"expected client.go, got {do['file']}"
        )
        assert do["line"] == 19, f"expected line 19, got {do['line']}"

    def test_e2e_caller_site(self, tools):
        """RunTest (test.go:19, package suite) calls r.ctrl.Publish from the
        e2e-tests directory.  The caller result must point at test.go line
        19 — cross-package from a test module."""
        line = self._publish_line(tools)
        callers = tools["get_callers"](CONTROLLER_FILE, line)
        rt = next((c for c in callers if c.get("caller") == "RunTest"), None)
        assert rt is not None, "RunTest not among callers"
        assert self._basename(rt["file"]) == "test.go", (
            f"expected test.go, got {rt['file']}"
        )
        assert rt["line"] == 19, f"expected line 19, got {rt['line']}"

    def test_callers_have_required_keys(self, tools):
        """Every caller dict must have file, line, name, caller, context."""
        line = self._publish_line(tools)
        callers = tools["get_callers"](CONTROLLER_FILE, line)
        for c in callers:
            for field in ("file", "line", "name", "caller", "context"):
                assert field in c, f"missing field {field!r} in caller {c!r}"
            assert c["line"] >= 1, f"caller line must be >= 1, got {c['line']}"

    def test_definition_itself_is_excluded(self, tools):
        """The definition line of Publish must NOT appear as a caller
        self-reference."""
        line = self._publish_line(tools)
        callers = tools["get_callers"](CONTROLLER_FILE, line)
        self_refs = [
            c for c in callers if c["file"] == CONTROLLER_FILE and c["line"] == line
        ]
        assert self_refs == [], f"definition line leaked into callers: {self_refs!r}"

    def test_caller_name_is_the_symbol(self, tools):
        """The `name` field of each caller result is the referenced symbol
        (Publish), not the caller function name."""
        line = self._publish_line(tools)
        callers = tools["get_callers"](CONTROLLER_FILE, line)
        for c in callers:
            assert c["name"] == "Publish", f"expected name 'Publish', got {c['name']!r}"

    def test_unused_func_has_no_callers(self, tools):
        """UnusedFunc (context.go) is never called — its callers list must
        be empty."""
        symbols = tools["get_file_outline"](SHARED_FILE)
        uf = next(s for s in symbols if s["name"] == "UnusedFunc")
        callers = tools["get_callers"](SHARED_FILE, uf["line"])
        assert callers == [], f"expected no callers for UnusedFunc, got {callers!r}"


# set_language_server_root rebuilds gopls at the new root
#
# Mirrors the Go simple and Python suites. Exercises the public
# set_language_server_root tool wrapper (the closure registered by
# make_tools), not tracer.set_root directly — covering the tool surface end
# to end. In a monorepo this is how to pick up an external go.work change or
# re-point gopls at a different module root. Re-rooting to a directory that
# does not contain the monorepo symbols makes them unreachable; clearing the
# root-keyed symbol cache and steering back makes them reachable again,
# proving the server genuinely rebuilds against the new root rather than the
# call being a no-op.
class TestSetLanguageServerRoot:
    """Steering the language server root rebuilds it at the new directory."""

    def test_changes_root_and_rebuilds_server(self, tools, tmp_path):
        # find_symbol works at the default monorepo root.
        before = tools["find_symbol"]("NewController")
        assert any(r["name"] == "NewController" and r.get("file") for r in before)

        # Steer at an empty directory: no .go files, so the symbol must not
        # be found. gopls needs a go.mod to index a module, so drop a
        # minimal one in the temp root.
        other = tmp_path / "other"
        other.mkdir()
        (other / "go.mod").write_text("module example.com/other\n\ngo 1.26.5\n")
        result = tools["set_language_server_root"](str(other))
        assert result["root"] == str(other)
        assert result["previous"] == str(MONOREPO_DIR)

        away = tools["find_symbol"]("NewController")
        # A real hit carries the symbol's file path; assert on that so the
        # check is uniform with the Python suite (whose "no symbols" hint
        # echoes the query name).
        assert not any(r.get("file") for r in away), (
            "NewController should not be found after steering at an empty "
            "root — the server did not rebuild against the new directory"
        )

        # Clear the root-keyed symbol cache so the next find_symbol must
        # re-query the rebuilt server instead of returning the cached hit.
        tools["cache"].clear_symbols()

        # Steering back to the monorepo root rebuilds the server there and
        # makes the symbol reachable again.
        tools["set_language_server_root"](str(MONOREPO_DIR))
        back = tools["find_symbol"]("NewController")
        assert any(r["name"] == "NewController" and r.get("file") for r in back), (
            "NewController should be found again after steering back — "
            "the server did not rebuild against the restored root"
        )
