"""Unit tests for Go contextual symbol search tools.

These tests use ``LocalShellBackend`` and run on the host without a
sandbox.  Sandbox/agent integration tests live in
``test_go_context_integration.py``.

The ``TestGotoDefinition`` class is the critical regression suite.  It
exercises the exact failure modes that were reported in
``goto_definition_bugfix_report.md`` and verified against a real gopls
v0.23.0 instance.  Every assertion encodes a *ground-truth* gopls
response, so a failure pinpoints the exact bug.

Ground truth (captured from `gopls definition -json` at the *member*
column of each call site in orders.go):

    orders.go line 40:  return strings.ToUpper(o.Process())
        strings.ToUpper @ col 17 -> strings.go:687   func strings.ToUpper(s string) string
        o.Process      @ col 27 -> orders.go:25      func (o *Order) Process() string

    orders.go line 26:  if !ValidateAddress(o.Address) {
        ValidateAddress @ col 6  -> validation.go:7   func ValidateAddress(address string) bool
        o.Address       @ col 24 -> orders.go:7        field Address string

    orders.go line 29:  formatted := FormatCurrency(o.Amount)
        FormatCurrency @ col 15 -> utils.go:6         func FormatCurrency(amount float64) string
        o.Amount       @ col 32 -> orders.go:8        field Amount float64

Key insight: for a selector expression ``pkg.Func`` / ``recv.Method`` /
``obj.Field``, gopls must be queried at the column of the **member**
(the part after the ``.``), NOT the qualifier/receiver or the dot.  If
the column points at the qualifier or the dot, gopls resolves to the
package import, the variable declaration, or the receiver type instead
of the target symbol.
"""

import os
import shutil
from pathlib import Path

import pytest

from marketplace.plugins.context_tools.extension import get_code_tools

# Source-of-truth sample in the repo; the module fixture copies it to a temp
# dir and rebinds these globals to the copy so the tracer's cache (.metalgate/)
# and gopls root land in temp, not in the committed sample.
SAMPLE_SRC = Path(__file__).parent / "sample" / "go" / "simple"
SAMPLE_DIR = SAMPLE_SRC
ORDERS_FILE = str(SAMPLE_SRC / "orders.go")
VALIDATION_FILE = str(SAMPLE_SRC / "validation.go")
UTILS_FILE = str(SAMPLE_SRC / "utils.go")
PROCESSOR_FILE = str(SAMPLE_SRC / "processor.go")
MULTISELECTOR_FILE = str(SAMPLE_SRC / "multiselector.go")
EXTERNAL_FILE = str(SAMPLE_SRC / "external.go")


@pytest.fixture(scope="module")
def tools(tmp_path_factory):
    global SAMPLE_DIR, ORDERS_FILE, VALIDATION_FILE, UTILS_FILE
    global PROCESSOR_FILE, MULTISELECTOR_FILE, EXTERNAL_FILE

    sample_dir = tmp_path_factory.mktemp("go_sample") / "simple"
    shutil.copytree(SAMPLE_SRC, sample_dir)
    SAMPLE_DIR = sample_dir
    ORDERS_FILE = str(sample_dir / "orders.go")
    VALIDATION_FILE = str(sample_dir / "validation.go")
    UTILS_FILE = str(sample_dir / "utils.go")
    PROCESSOR_FILE = str(sample_dir / "processor.go")
    MULTISELECTOR_FILE = str(sample_dir / "multiselector.go")
    EXTERNAL_FILE = str(sample_dir / "external.go")

    tool_list = get_code_tools(cwd=str(sample_dir), language="go")
    (
        goto_def,
        outline,
        get_source,
        callers,
        callees,
        find_sym,
        set_root,
    ) = tool_list

    yield {
        "goto_definition": goto_def,
        "get_file_outline": outline,
        "get_source": get_source,
        "get_callers": callers,
        "get_callees": callees,
        "find_symbol": find_sym,
        "set_language_server_root": set_root,
    }


# get_file_outline
class TestGetFileOutline:
    def test_finds_struct(self, tools):
        symbols = tools["get_file_outline"](ORDERS_FILE)
        assert any(s["name"] == "Order" and s["kind"] == "struct" for s in symbols)

    def test_finds_interface(self, tools):
        symbols = tools["get_file_outline"](ORDERS_FILE)
        assert any(
            s["name"] == "Processor" and s["kind"] == "interface" for s in symbols
        )

    def test_finds_function(self, tools):
        symbols = tools["get_file_outline"](ORDERS_FILE)
        assert any(s["name"] == "NewOrder" and s["kind"] == "function" for s in symbols)

    def test_finds_method(self, tools):
        symbols = tools["get_file_outline"](ORDERS_FILE)
        method = next(
            (s for s in symbols if s["name"] == "Process" and s["kind"] == "method"),
            None,
        )
        assert method is not None

    def test_method_has_receiver(self, tools):
        symbols = tools["get_file_outline"](ORDERS_FILE)
        process = next(s for s in symbols if s["name"] == "Process")
        assert "Order" in (process.get("class") or "")

    def test_signature_contains_name(self, tools):
        symbols = tools["get_file_outline"](VALIDATION_FILE)
        validate = next(s for s in symbols if s["name"] == "ValidateAddress")
        assert "ValidateAddress" in validate["signature"]

    def test_cached_result_is_identical(self, tools):
        first = tools["get_file_outline"](ORDERS_FILE)
        second = tools["get_file_outline"](ORDERS_FILE)
        assert first == second


# get_source
class TestGetSource:
    def _validate_line(self, tools):
        symbols = tools["get_file_outline"](VALIDATION_FILE)
        return next(s for s in symbols if s["name"] == "ValidateAddress")["line"]

    def test_source_contains_func(self, tools):
        line = self._validate_line(tools)
        result = tools["get_source"](VALIDATION_FILE, line)
        assert "ValidateAddress" in result["source"]
        assert "return false" in result["source"]

    def test_start_and_end_lines_are_sane(self, tools):
        line = self._validate_line(tools)
        result = tools["get_source"](VALIDATION_FILE, line)
        assert result["start_line"] >= 1
        assert result["end_line"] >= result["start_line"]

    def test_get_source_for_struct(self, tools):
        symbols = tools["get_file_outline"](ORDERS_FILE)
        st = next(s for s in symbols if s["name"] == "Order")
        result = tools["get_source"](ORDERS_FILE, st["line"])
        assert "type Order struct" in result["source"]

    def test_get_source_from_body_line(self, tools):
        """get_source should work when given any line inside the function body."""
        symbols = tools["get_file_outline"](VALIDATION_FILE)
        va = next(s for s in symbols if s["name"] == "ValidateAddress")
        body_line = va["line"] + 2  # inside the body
        result = tools["get_source"](VALIDATION_FILE, body_line)
        assert "ValidateAddress" in result["source"]

    def test_fallback_context_window(self, tools):
        result = tools["get_source"](VALIDATION_FILE, 1, context=10)
        assert isinstance(result["source"], str)

    def test_nonexistent_file_returns_error(self, tools):
        result = tools["get_source"]("/nonexistent/file.go", 1)
        assert result["source"] == "" or "error" in result


# find_symbol
class TestFindSymbol:
    def test_exact_match_finds_validate_address(self, tools):
        results = tools["find_symbol"]("ValidateAddress")
        names = [r["name"] for r in results]
        assert "ValidateAddress" in names

    def test_exact_match_does_not_find_partial(self, tools):
        results = tools["find_symbol"]("Validate")
        names = [r["name"] for r in results]
        assert "ValidateAddress" not in names

    def test_case_insensitive(self, tools):
        results = tools["find_symbol"]("validateaddress")
        names = [r["name"] for r in results]
        assert "ValidateAddress" in names

    def test_unknown_symbol_returns_empty_list(self, tools):
        results = tools["find_symbol"]("zzz_does_not_exist_xyz")
        assert results == []

    def test_finds_struct_by_name(self, tools):
        results = tools["find_symbol"]("Order")
        names = [r["name"] for r in results]
        assert "Order" in names

    def test_cached_result_is_identical(self, tools):
        first = tools["find_symbol"]("ValidateAddress")
        second = tools["find_symbol"]("ValidateAddress")
        assert first == second

    def test_results_scoped_to_project_root(self, tools):
        """find_symbol should only return symbols within the project root,
        not from other projects, stdlib, or the module cache."""
        results = tools["find_symbol"]("Order")
        for r in results:
            assert str(SAMPLE_DIR) in r["file"], (
                f"Result file {r['file']} is outside project root {SAMPLE_DIR}"
            )


# goto_definition
#
# This is the critical regression suite.  Each test resolves a specific
# call site in orders.go and asserts the EXACT location/kind/signature that
# a real gopls returns when queried at the member column.
#
# The call sites (all in orders.go):
#
#   line 40:  return strings.ToUpper(o.Process())
#   line 26:  if !ValidateAddress(o.Address) {
#   line 29:  formatted := FormatCurrency(o.Amount)
#
# Ground truth captured from `gopls definition -json` at the member column:
#
#   strings.ToUpper  -> strings.go:687  "func strings.ToUpper(s string) string"
#   o.Process        -> orders.go:25    "func (o *Order) Process() string"
#   ValidateAddress  -> validation.go:7  "func ValidateAddress(address string) bool"
#   o.Address        -> orders.go:7      "field Address string"
#   FormatCurrency   -> utils.go:6       "func FormatCurrency(amount float64) string"
#   o.Amount         -> orders.go:8      "field Amount float64"
#
# The bug under test: for selector expressions (pkg.Func, recv.Method,
# obj.Field) the tool sends gopls the column of the qualifier/receiver or
# the dot instead of the member.  gopls then resolves to the import line,
# the variable declaration, or the receiver type — NOT the target symbol.
# Every test below fails when that bug is present and passes once the
# column computation is fixed to point at the member.


class TestGotoDefinition:
    """Resolves call sites in orders.go and checks against gopls ground truth.

    All call sites live in orders.go lines 26, 29, and 40:

        26:  if !ValidateAddress(o.Address) {
        29:  formatted := FormatCurrency(o.Amount)
        40:  return strings.ToUpper(o.Process())
    """

    # helpers

    @staticmethod
    def _basename(path: str) -> str:
        return path.replace("\\", "/").rstrip("/").split("/")[-1]

    # stdlib qualified call: strings.ToUpper

    def test_resolves_qualified_stdlib_call(self, tools):
        """strings.ToUpper (orders.go:40) must resolve to the stdlib function,
        not the `import "strings"` line.

        gopls ground truth at the member column (col 17, the 'T' of ToUpper):
            strings.go:668  func strings.ToUpper(s string) string
        """
        result = tools["goto_definition"](ORDERS_FILE, 40, "strings.ToUpper")
        assert self._basename(result["file"]) == "strings.go", (
            f"expected strings.go, got {result['file']} "
            "(resolving the package import instead of the function)"
        )
        assert result["line"] == 668, f"expected line 668, got {result['line']}"
        assert "ToUpper" in result["signature"], (
            f"expected signature containing 'ToUpper', got {result['signature']!r}"
        )

    def test_qualified_stdlib_call_kind_is_function(self, tools):
        """strings.ToUpper must be classified as a function, not 'unknown'."""
        result = tools["goto_definition"](ORDERS_FILE, 40, "strings.ToUpper")
        assert result["kind"] == "function", (
            f"expected kind 'function', got {result['kind']!r} "
            "(kind inference is falling back to 'unknown' because the hover "
            "signature is the package docstring, not the function signature)"
        )

    def test_qualified_stdlib_call_not_import_line(self, tools):
        """The result must NOT be the `import "strings"` line in orders.go.

        This is a regression guard for the column bug: when the column points
        at the qualifier `strings` (or the dot), gopls returns the import
        statement with signature 'package strings'.  The result file would be
        orders.go and the line would be 3 (the import line).
        """
        result = tools["goto_definition"](ORDERS_FILE, 40, "strings.ToUpper")
        assert not (
            self._basename(result["file"]) == "orders.go" and result["line"] == 3
        ), (
            "resolved to the import line — column points at the qualifier, "
            "not the member"
        )

    # concrete method call: o.Process

    def test_resolves_method_call_on_receiver(self, tools):
        """o.Process (orders.go:40) must resolve to the method definition,
        not the `var o *Order` declaration or the receiver type.

        gopls ground truth at the member column (col 27, the 'P' of Process):
            orders.go:25  func (o *Order) Process() string
        """
        result = tools["goto_definition"](ORDERS_FILE, 40, "o.Process")
        assert self._basename(result["file"]) == "orders.go", (
            f"expected orders.go, got {result['file']}"
        )
        assert result["line"] == 25, f"expected line 25, got {result['line']}"
        assert "Process" in result["signature"], (
            f"expected signature containing 'Process', got {result['signature']!r}"
        )

    def test_method_call_kind_is_method(self, tools):
        """o.Process must be classified as a method, not 'unknown'."""
        result = tools["goto_definition"](ORDERS_FILE, 40, "o.Process")
        assert result["kind"] == "method", (
            f"expected kind 'method', got {result['kind']!r}"
        )

    def test_method_call_not_receiver_declaration(self, tools):
        """The result must NOT be the receiver variable declaration.

        Regression guard: when the column points at the receiver `o` (or the
        dot), gopls returns `var o *Order` at the function signature line
        (line 25 in older gopls, or the parameter line).  The signature would
        be 'var o *Order' instead of the method signature.
        """
        result = tools["goto_definition"](ORDERS_FILE, 40, "o.Process")
        assert "var " not in result["signature"], (
            f"resolved to a variable declaration: {result['signature']!r} "
            "(column points at the receiver, not the member)"
        )

    # plain function call: ValidateAddress

    def test_resolves_plain_function_call(self, tools):
        """ValidateAddress (orders.go:26) must resolve to validation.go.

        gopls ground truth at col 6:
            validation.go:7  func ValidateAddress(address string) bool
        """
        result = tools["goto_definition"](ORDERS_FILE, 26, "ValidateAddress")
        assert self._basename(result["file"]) == "validation.go", (
            f"expected validation.go, got {result['file']}"
        )
        assert result["line"] == 7, f"expected line 7, got {result['line']}"
        assert "ValidateAddress" in result["signature"]

    def test_plain_function_call_kind_is_function(self, tools):
        result = tools["goto_definition"](ORDERS_FILE, 26, "ValidateAddress")
        assert result["kind"] == "function", (
            f"expected kind 'function', got {result['kind']!r}"
        )

    # plain function call: FormatCurrency

    def test_resolves_plain_function_call_other_file(self, tools):
        """FormatCurrency (orders.go:29) must resolve to utils.go.

        gopls ground truth at col 15:
            utils.go:6  func FormatCurrency(amount float64) string
        """
        result = tools["goto_definition"](ORDERS_FILE, 29, "FormatCurrency")
        assert self._basename(result["file"]) == "utils.go", (
            f"expected utils.go, got {result['file']}"
        )
        assert result["line"] == 6, f"expected line 6, got {result['line']}"
        assert "FormatCurrency" in result["signature"]

    def test_plain_function_call_other_file_kind_is_function(self, tools):
        result = tools["goto_definition"](ORDERS_FILE, 29, "FormatCurrency")
        assert result["kind"] == "function", (
            f"expected kind 'function', got {result['kind']!r}"
        )

    # struct field access: o.Address

    def test_resolves_struct_field_access(self, tools):
        """o.Address (orders.go:26) must resolve to the struct field,
        not the `var o *Order` declaration.

        gopls ground truth at the member column (col 24, the 'A' of Address):
            orders.go:7  field Address string
        """
        result = tools["goto_definition"](ORDERS_FILE, 26, "o.Address")
        assert self._basename(result["file"]) == "orders.go", (
            f"expected orders.go, got {result['file']}"
        )
        assert result["line"] == 7, f"expected line 7, got {result['line']}"
        assert "Address" in result["signature"], (
            f"expected signature containing 'Address', got {result['signature']!r}"
        )

    def test_struct_field_access_not_receiver_declaration(self, tools):
        """o.Address must NOT resolve to `var o *Order`.

        Regression guard: when the column points at the receiver `o`, gopls
        returns the variable declaration `var o *Order` at line 25.
        """
        result = tools["goto_definition"](ORDERS_FILE, 26, "o.Address")
        assert "var " not in result["signature"], (
            f"resolved to a variable declaration: {result['signature']!r} "
            "(column points at the receiver, not the member)"
        )

    # struct field access: o.Amount

    def test_resolves_struct_field_access_amount(self, tools):
        """o.Amount (orders.go:29) must resolve to the struct field.

        gopls ground truth at the member column (col 32, the 'A' of Amount):
            orders.go:8  field Amount float64
        """
        result = tools["goto_definition"](ORDERS_FILE, 29, "o.Amount")
        assert self._basename(result["file"]) == "orders.go", (
            f"expected orders.go, got {result['file']}"
        )
        assert result["line"] == 8, f"expected line 8, got {result['line']}"
        assert "Amount" in result["signature"], (
            f"expected signature containing 'Amount', got {result['signature']!r}"
        )

    def test_struct_field_access_amount_not_receiver_declaration(self, tools):
        result = tools["goto_definition"](ORDERS_FILE, 29, "o.Amount")
        assert "var " not in result["signature"], (
            f"resolved to a variable declaration: {result['signature']!r} "
            "(column points at the receiver, not the member)"
        )

    # interface method call: p.Process
    #
    # processor.go defines UseProcessor(p Processor) which calls p.Process().
    # gopls ground truth at the member column (col 11, the 'P' of Process):
    #     orders.go:13  func (Processor) Process() string
    # (the interface method line inside `type Processor interface`)

    def test_resolves_interface_method_call(self, tools):
        """p.Process (where p is of interface type Processor) must resolve
        to the method, not the interface type or the parameter declaration.

        gopls ground truth at the member column:
            orders.go:13  func (Processor) Process() string
        (the interface method line inside `type Processor interface`)

        Note: some gopls versions resolve interface-method calls to the
        `type Processor interface` line (line 12) rather than the method
        line (line 13).  Both are acceptable as long as the result is NOT
        the parameter declaration `var p Processor` and the kind is
        'method'.
        """
        result = tools["goto_definition"](PROCESSOR_FILE, 5, "p.Process")
        # Must resolve into orders.go (where the interface is defined),
        # not back to the parameter declaration in processor.go.
        assert self._basename(result["file"]) == "orders.go", (
            f"expected orders.go, got {result['file']} "
            "(resolving the parameter declaration instead of the method)"
        )
        assert result["line"] in (12, 13), (
            f"expected line 12 or 13, got {result['line']}"
        )
        assert result["kind"] == "method", (
            f"expected kind 'method', got {result['kind']!r}"
        )

    def test_interface_method_call_not_parameter_declaration(self, tools):
        """p.Process must NOT resolve to `var p Processor`."""
        result = tools["goto_definition"](PROCESSOR_FILE, 5, "p.Process")
        assert "var " not in result["signature"], (
            f"resolved to a variable declaration: {result['signature']!r} "
            "(column points at the receiver, not the member)"
        )

    # same-line disambiguation
    #
    # orders.go:40 has TWO selectors on the same line:
    #     return strings.ToUpper(o.Process())
    # A correct implementation must resolve each one independently based on
    # the name argument, not blindly pick the first selector on the line.

    def test_same_line_disambiguation_first_selector(self, tools):
        """On line 40, `strings.ToUpper` must resolve to strings.go (the
        first selector), not orders.go (the second selector's method)."""
        result = tools["goto_definition"](ORDERS_FILE, 40, "strings.ToUpper")
        assert self._basename(result["file"]) == "strings.go", (
            f"expected strings.go, got {result['file']}"
        )

    def test_same_line_disambiguation_second_selector(self, tools):
        """On line 40, `o.Process` must resolve to orders.go:25 (the second
        selector), not strings.go (the first selector)."""
        result = tools["goto_definition"](ORDERS_FILE, 40, "o.Process")
        assert self._basename(result["file"]) == "orders.go", (
            f"expected orders.go, got {result['file']}"
        )
        assert result["line"] == 25, f"expected line 25, got {result['line']}"

    # result shape

    def test_result_has_required_fields(self, tools):
        """Every result must include name, kind, file, line, col, signature."""
        result = tools["goto_definition"](ORDERS_FILE, 26, "ValidateAddress")
        for field in ("name", "kind", "file", "line", "col", "signature"):
            assert field in result, f"missing field {field!r} in result"

    def test_col_is_member_column_for_selector(self, tools):
        """For a selector expression, the returned `col` must point at the
        member (the part after the dot), not the qualifier or the dot.

        For strings.ToUpper on line 40, the member 'ToUpper' starts at
        column 17 (1-indexed).  The returned col must be >= 17 and point
        within the 'ToUpper' token, i.e. in the range [17, 23].
        """
        result = tools["goto_definition"](ORDERS_FILE, 40, "strings.ToUpper")
        assert 17 <= result["col"] <= 23, (
            f"expected col in [17, 23] (the 'ToUpper' member), "
            f"got col={result['col']} "
            "(column points at the qualifier or dot, not the member)"
        )

    def test_col_is_member_column_for_method(self, tools):
        """For o.Process on line 40, the member 'Process' starts at column 27.
        The returned col must be in [27, 34]."""
        result = tools["goto_definition"](ORDERS_FILE, 40, "o.Process")
        assert 27 <= result["col"] <= 34, (
            f"expected col in [27, 34] (the 'Process' member), got col={result['col']}"
        )

    def test_col_is_member_column_for_field(self, tools):
        """For o.Address on line 26, the member 'Address' starts at column 24.
        The returned col must be in [24, 31]."""
        result = tools["goto_definition"](ORDERS_FILE, 26, "o.Address")
        assert 24 <= result["col"] <= 31, (
            f"expected col in [24, 31] (the 'Address' member), got col={result['col']}"
        )


# get_goto_definition — same qualifier appearing twice on one line
#
# This is the pattern that the gin codebase exercises but the original test
# suite missed.  In gin's recovery.go:56:
#
#     logger = log.New(out, "\n\n\x1b[31m", log.LstdFlags)
#
# the qualifier `log` appears twice on the same line (log.New and
# log.LstdFlags).  A column computation that finds the *first* occurrence of
# the qualifier, or the *first* dot on the line, will resolve the wrong
# symbol.  The tool must use the `name` argument to find the exact
# `qualifier.member` pair and compute the column of *that* member.
#
# multiselector.go:8 has the same shape with `strings` as the qualifier:
#
#     return strings.ToUpper(strings.ToLower(s))
#
# gopls ground truth (captured from gopls v0.23.0):
#     strings.ToUpper @ col 17 -> strings.go:687  func strings.ToUpper(s string) string
#     strings.ToLower @ col 33 -> strings.go:727  func strings.ToLower(s string) string
class TestGotoDefinitionSameQualifierTwice:
    """Resolves two selectors that share the same qualifier on one line.

    multiselector.go line 8:
        return strings.ToUpper(strings.ToLower(s))

    Both selectors use the qualifier `strings`.  The tool must distinguish
    them by the member name in the `name` argument, not by finding the
    first `strings.` on the line.
    """

    @staticmethod
    def _basename(path: str) -> str:
        return path.replace("\\", "/").rstrip("/").split("/")[-1]

    def test_first_occurrence_resolves_correctly(self, tools):
        """strings.ToUpper (the first `strings.` on line 8) must resolve to
        strings.go:668, not the import line."""
        result = tools["goto_definition"](MULTISELECTOR_FILE, 8, "strings.ToUpper")
        assert self._basename(result["file"]) == "strings.go", (
            f"expected strings.go, got {result['file']} "
            "(resolving the package import instead of the function)"
        )
        assert result["line"] == 668, f"expected line 668, got {result['line']}"
        assert "ToUpper" in result["signature"], (
            f"expected signature containing 'ToUpper', got {result['signature']!r}"
        )

    def test_second_occurrence_resolves_correctly(self, tools):
        """strings.ToLower (the SECOND `strings.` on line 8) must resolve to
        strings.go:708, not strings.go:668 (the first occurrence) and not
        the import line."""
        result = tools["goto_definition"](MULTISELECTOR_FILE, 8, "strings.ToLower")
        assert self._basename(result["file"]) == "strings.go", (
            f"expected strings.go, got {result['file']} "
            "(resolving the package import instead of the function)"
        )
        assert result["line"] == 708, (
            f"expected line 708 (ToLower), got {result['line']} "
            "(resolving the first occurrence ToUpper instead of the requested ToLower)"
        )
        assert "ToLower" in result["signature"], (
            f"expected signature containing 'ToLower', got {result['signature']!r}"
        )

    def test_second_occurrence_not_first(self, tools):
        """The result for strings.ToLower must NOT be the same as
        strings.ToUpper.  This catches a column computation that always
        picks the first selector on the line regardless of the name arg."""
        first = tools["goto_definition"](MULTISELECTOR_FILE, 8, "strings.ToUpper")
        second = tools["goto_definition"](MULTISELECTOR_FILE, 8, "strings.ToLower")
        assert first["line"] != second["line"], (
            f"both selectors resolved to the same line {first['line']} "
            "(column computation is picking the first selector, not the "
            "one matching the name argument)"
        )

    def test_second_occurrence_not_import_line(self, tools):
        """strings.ToLower must NOT resolve to the import line."""
        result = tools["goto_definition"](MULTISELECTOR_FILE, 8, "strings.ToLower")
        assert not (
            self._basename(result["file"]) == "multiselector.go" and result["line"] == 3
        ), (
            "resolved to the import line — column points at the qualifier, "
            "not the member"
        )

    def test_col_points_at_correct_member_first(self, tools):
        """For strings.ToUpper on line 8, col must be in [17, 24] (the
        'ToUpper' token), not at the qualifier or dot."""
        result = tools["goto_definition"](MULTISELECTOR_FILE, 8, "strings.ToUpper")
        assert 17 <= result["col"] <= 24, (
            f"expected col in [17, 24] (the 'ToUpper' member), got col={result['col']}"
        )

    def test_col_points_at_correct_member_second(self, tools):
        """For strings.ToLower on line 8, col must be in [33, 40] (the
        'ToLower' token), not at the first selector or the qualifier."""
        result = tools["goto_definition"](MULTISELECTOR_FILE, 8, "strings.ToLower")
        assert 33 <= result["col"] <= 40, (
            f"expected col in [33, 40] (the 'ToLower' member), "
            f"got col={result['col']} "
            "(column points at the first selector, not the requested one)"
        )


# 3rd-party package call
#
# external.go imports gin and calls gin.New().  This reproduces the exact
# bug reported in the gin codebase: a qualified call to a 3rd-party package
# resolves to the `import "github.com/gin-gonic/gin"` line (signature
# "package gin") instead of the actual function definition in the package.
#
# gopls ground truth (captured from gopls v0.23.0):
#     At the member column (col 13, the 'N' of New):
#         gin.go:202  func gin.New(opts ...gin.OptionFunc) *gin.Engine
#     At the dot (col 12):
#         external.go:3  package gin  (the import line — THIS IS THE BUG)
class TestGotoDefinitionThirdParty:
    """gin.New (external.go:7) must resolve to the function in the 3rd-party
    package, not the import line in the caller.

    external.go line 7:
        return gin.New()

    gopls ground truth at the member column:
        gin.go:202  func gin.New(opts ...gin.OptionFunc) *gin.Engine
    """

    @staticmethod
    def _basename(path: str) -> str:
        return path.replace("\\", "/").rstrip("/").split("/")[-1]

    def test_resolves_third_party_function(self, tools):
        """gin.New must resolve to gin.go (the 3rd-party package source),
        not external.go:3 (the import line in the caller)."""
        result = tools["goto_definition"](EXTERNAL_FILE, 7, "gin.New")
        assert result, "got empty result"
        assert self._basename(result["file"]) == "gin.go", (
            f"expected gin.go (the function definition), "
            f"got {result['file']} "
            "(resolving the package import instead of the function)"
        )
        assert "New" in result["signature"], (
            f"expected signature containing 'New', "
            f"got {result['signature']!r} "
            "(got the package docstring instead of the function signature)"
        )

    def test_third_party_not_import_line(self, tools):
        """The result must NOT be the import line in external.go.

        When the column points at the dot or qualifier, gopls returns the
        import statement: external.go:3 with signature 'package gin'.
        """
        result = tools["goto_definition"](EXTERNAL_FILE, 7, "gin.New")
        assert not (
            self._basename(result["file"]) == "external.go" and result["line"] == 3
        ), (
            "resolved to the import line — the column points at the "
            "qualifier/dot, not the member 'New'"
        )

    def test_third_party_kind_is_function(self, tools):
        """gin.New must be classified as a function, not 'unknown'."""
        result = tools["goto_definition"](EXTERNAL_FILE, 7, "gin.New")
        assert result["kind"] == "function", (
            f"expected kind 'function', got {result['kind']!r} "
            "(kind inference fell back to 'unknown' because the hover "
            "signature is the package docstring, not the function signature)"
        )


# get_callees
#
# Process (orders.go:25) is the cross-file hub: it calls ValidateAddress
# (validation.go) and FormatCurrency (utils.go).  gopls ground truth at the
# member column of each call site:
#     ValidateAddress  -> validation.go:7   func ValidateAddress(address string) bool
#     FormatCurrency   -> utils.go:6        func FormatCurrency(amount float64) string
#
# get_callees uses tree-sitter to find call positions inside the function
# body, then resolves each via textDocument/definition.  Every callee must
# resolve across files with name, file, line, kind, and signature.
class TestGetCallees:
    """Lists every symbol called by the function on `line` of `file`.

    Process (orders.go:25) calls ValidateAddress and FormatCurrency, both
    defined in other files.  Each result must resolve to the callee's
    definition, not the call site.
    """

    @staticmethod
    def _basename(path: str) -> str:
        return path.replace("\\", "/").rstrip("/").split("/")[-1]

    def _process_line(self, tools):
        symbols = tools["get_file_outline"](ORDERS_FILE)
        return next(s for s in symbols if s["name"] == "Process")["line"]

    def test_finds_validate_address_and_format_currency(self, tools):
        """Process (orders.go:25) must list ValidateAddress and FormatCurrency
        as callees.

        gopls ground truth:
            ValidateAddress -> validation.go:7
            FormatCurrency  -> utils.go:6
        """
        line = self._process_line(tools)
        callees = tools["get_callees"](ORDERS_FILE, line)
        names = [c["name"] for c in callees]
        assert "ValidateAddress" in names, (
            f"expected ValidateAddress among callees, got {names!r}"
        )
        assert "FormatCurrency" in names, (
            f"expected FormatCurrency among callees, got {names!r}"
        )

    def test_callees_cross_file(self, tools):
        """At least one callee must resolve to a different file than the
        caller's file — gopls must index the sibling files of the package."""
        line = self._process_line(tools)
        callees = tools["get_callees"](ORDERS_FILE, line)
        files = [c["file"] for c in callees]
        assert any("orders.go" not in f for f in files), (
            f"no callee resolved outside orders.go; got {files!r}"
        )

    def test_validate_address_callee_resolves_to_definition(self, tools):
        """The ValidateAddress callee must point at validation.go:7, the
        definition — not the call site in orders.go."""
        line = self._process_line(tools)
        callees = tools["get_callees"](ORDERS_FILE, line)
        va = next((c for c in callees if c["name"] == "ValidateAddress"), None)
        assert va is not None, "ValidateAddress not in callees"
        assert self._basename(va["file"]) == "validation.go", (
            f"expected validation.go, got {va['file']}"
        )
        assert va["line"] == 7, f"expected line 7, got {va['line']}"

    def test_format_currency_callee_resolves_to_definition(self, tools):
        """The FormatCurrency callee must point at utils.go:6."""
        line = self._process_line(tools)
        callees = tools["get_callees"](ORDERS_FILE, line)
        fc = next((c for c in callees if c["name"] == "FormatCurrency"), None)
        assert fc is not None, "FormatCurrency not in callees"
        assert self._basename(fc["file"]) == "utils.go", (
            f"expected utils.go, got {fc['file']}"
        )
        assert fc["line"] == 6, f"expected line 6, got {fc['line']}"

    def test_callees_have_required_keys(self, tools):
        """Every callee dict must have name, kind, file, line, signature."""
        line = self._process_line(tools)
        callees = tools["get_callees"](ORDERS_FILE, line)
        for c in callees:
            for field in ("name", "kind", "file", "line", "signature"):
                assert field in c, f"missing field {field!r} in callee {c!r}"
            assert c["line"] >= 1, f"callee line must be >= 1, got {c['line']}"

    def test_validate_address_callee_kind_is_function(self, tools):
        """ValidateAddress must be classified as a function."""
        line = self._process_line(tools)
        callees = tools["get_callees"](ORDERS_FILE, line)
        va = next((c for c in callees if c["name"] == "ValidateAddress"), None)
        assert va is not None, "ValidateAddress not in callees"
        assert va["kind"] == "function", (
            f"expected kind 'function', got {va['kind']!r}"
        )

    def test_no_callees_for_leaf_function(self, tools):
        """FormatCurrency (utils.go:6) only calls fmt.Sprintf (stdlib, outside
        the project) — its project callees list is empty.

        gopls resolves fmt.Sprintf to the stdlib; get_callees reads the
        definition outline and, for stdlib paths, hovers for the signature.
        The result must still be a list (possibly empty or stdlib-only),
        never None and never raising."""
        symbols = tools["get_file_outline"](UTILS_FILE)
        fc = next(s for s in symbols if s["name"] == "FormatCurrency")
        callees = tools["get_callees"](UTILS_FILE, fc["line"])
        assert isinstance(callees, list), f"expected list, got {type(callees)}"
        # No project-defined symbol is called by FormatCurrency.
        project_names = [c["name"] for c in callees if c.get("name")]
        assert "ValidateAddress" not in project_names
        assert "FormatCurrency" not in project_names


# get_callers
#
# Process (orders.go:25) is called from three sites across two files:
#     main.go:7          fmt.Println(o.Process())     caller: main
#     orders.go:40       return strings.ToUpper(o.Process())  caller: ProcessAndUpper
#     processor.go:5     return p.Process()            caller: UseProcessor
#
# get_callers uses LSP call hierarchy (prepareCallHierarchy + incomingCalls).
# Each result points at the actual call site with file, line, caller (the
# enclosing function), and context (the referencing line).  The definition's
# own line must never appear as a self-reference.
class TestGetCallers:
    """Finds every site that calls the symbol defined on `line` of `file`.

    Process (orders.go:25) is called from main, ProcessAndUpper, and
    UseProcessor.  Each result must point at the call site, not the
    definition, and the definition line must be excluded.
    """

    @staticmethod
    def _basename(path: str) -> str:
        return path.replace("\\", "/").rstrip("/").split("/")[-1]

    def _process_line(self, tools):
        symbols = tools["get_file_outline"](ORDERS_FILE)
        return next(s for s in symbols if s["name"] == "Process")["line"]

    def test_finds_three_callers(self, tools):
        """Process must have at least three callers: main, ProcessAndUpper,
        and UseProcessor."""
        line = self._process_line(tools)
        callers = tools["get_callers"](ORDERS_FILE, line)
        callers_by_name = {c.get("caller", "") for c in callers}
        for expected in ("main", "ProcessAndUpper", "UseProcessor"):
            assert expected in callers_by_name, (
                f"expected caller {expected!r}, got {callers_by_name!r}"
            )

    def test_main_caller_site(self, tools):
        """main (main.go:7) calls o.Process().  The caller result must point
        at main.go line 7."""
        line = self._process_line(tools)
        callers = tools["get_callers"](ORDERS_FILE, line)
        main_caller = next(
            (c for c in callers if c.get("caller") == "main"), None
        )
        assert main_caller is not None, "main not among callers"
        assert self._basename(main_caller["file"]) == "main.go", (
            f"expected main.go, got {main_caller['file']}"
        )
        assert main_caller["line"] == 7, (
            f"expected line 7, got {main_caller['line']}"
        )

    def test_callers_have_required_keys(self, tools):
        """Every caller dict must have file, line, name, caller, context."""
        line = self._process_line(tools)
        callers = tools["get_callers"](ORDERS_FILE, line)
        for c in callers:
            for field in ("file", "line", "name", "caller", "context"):
                assert field in c, f"missing field {field!r} in caller {c!r}"
            assert c["line"] >= 1, f"caller line must be >= 1, got {c['line']}"

    def test_definition_itself_is_excluded(self, tools):
        """The definition line of Process must NOT appear as a caller
        self-reference."""
        line = self._process_line(tools)
        callers = tools["get_callers"](ORDERS_FILE, line)
        self_refs = [
            c for c in callers if c["file"] == ORDERS_FILE and c["line"] == line
        ]
        assert self_refs == [], (
            f"definition line leaked into callers: {self_refs!r}"
        )

    def test_caller_name_is_the_symbol(self, tools):
        """The `name` field of each caller result is the referenced symbol
        (Process), not the caller function name."""
        line = self._process_line(tools)
        callers = tools["get_callers"](ORDERS_FILE, line)
        for c in callers:
            assert c["name"] == "Process", (
                f"expected name 'Process', got {c['name']!r}"
            )

    def test_unused_func_has_no_callers(self, tools):
        """UnusedFunc (orders.go) is never called — its callers list must be
        empty (no note entry, unlike find_symbol)."""
        symbols = tools["get_file_outline"](ORDERS_FILE)
        uf = next(s for s in symbols if s["name"] == "UnusedFunc")
        callers = tools["get_callers"](ORDERS_FILE, uf["line"])
        assert callers == [], f"expected no callers for UnusedFunc, got {callers!r}"


# relative paths
#
# Every public tool that takes a `file` arg must accept a bare filename
# (relative path) and resolve it against the tracer root, not the process
# CWD.  Before the fix, the first action of each tool was
# `self._read_file(file)` / `self._to_host_uri(file)` using the raw path,
# which resolved against the Python CWD — a directory the language server
# was never pointed at — so `read_text` raised FileNotFoundError before any
# LSP request was sent.
#
# The fix normalizes once at each public entry (Tracer._resolve_path), so
# reads, URIs, cache keys, and the `file` field in returned dicts all agree.
# These tests pass a bare filename and assert the result matches the
# absolute-path result exactly.  They also assert CWD is NOT the sample dir,
# so a fix that resolves against CWD instead of self.root would fail here.
class TestRelativePaths:
    """Relative `file` args resolve against the tracer root, not the CWD.

    For each tool, calling with the bare filename of ORDERS_FILE must produce
    the same result as calling with the absolute path — same names, lines,
    and files — and must not raise.
    """

    @staticmethod
    def _basename(path: str) -> str:
        return path.replace("\\", "/").rstrip("/").split("/")[-1]

    def test_cwd_is_not_sample_dir(self, tools):
        """Guard: the process CWD must not be the sample dir, so a fix that
        resolves relative paths against CWD (instead of self.root) would
        fail the rest of this class."""
        assert os.getcwd() != str(SAMPLE_DIR), (
            "CWD equals the sample dir — the relative-path tests cannot "
            "distinguish a self.root-based fix from a CWD-based one"
        )

    def test_get_file_outline_matches_absolute(self, tools):
        rel = Path(ORDERS_FILE).name
        rel_result = tools["get_file_outline"](rel)
        abs_result = tools["get_file_outline"](ORDERS_FILE)
        assert rel_result, "relative-path outline returned empty"
        assert rel_result == abs_result, (
            "relative-path outline differs from absolute-path outline"
        )

    def test_get_file_outline_no_exception_on_relative(self, tools):
        rel = Path(ORDERS_FILE).name
        # Must not raise FileNotFoundError (the reported bug).
        result = tools["get_file_outline"](rel)
        assert isinstance(result, list)

    def test_goto_definition_matches_absolute(self, tools):
        rel = Path(ORDERS_FILE).name
        rel_result = tools["goto_definition"](rel, 40, "o.Process")
        abs_result = tools["goto_definition"](ORDERS_FILE, 40, "o.Process")
        assert rel_result, "relative-path goto_definition returned empty"
        assert rel_result == abs_result, (
            "relative-path goto_definition differs from absolute-path"
        )

    def test_goto_definition_no_exception_on_relative(self, tools):
        rel = Path(ORDERS_FILE).name
        # Must not raise FileNotFoundError (the reported bug).
        result = tools["goto_definition"](rel, 40, "o.Process")
        assert result is not None

    def test_get_source_matches_absolute(self, tools):
        symbols = tools["get_file_outline"](ORDERS_FILE)
        line = next(s for s in symbols if s["name"] == "Process")["line"]
        rel = Path(ORDERS_FILE).name
        rel_result = tools["get_source"](rel, line)
        abs_result = tools["get_source"](ORDERS_FILE, line)
        assert rel_result["source"], "relative-path get_source returned empty"
        assert rel_result == abs_result, (
            "relative-path get_source differs from absolute-path"
        )
        # The file field must be the normalized absolute path, not the bare
        # filename — reads, URIs, and the returned `file` must all agree.
        assert rel_result["file"] == ORDERS_FILE, (
            f"get_source file field is {rel_result['file']!r}, expected "
            f"{ORDERS_FILE!r} (the normalized absolute path)"
        )

    def test_get_source_no_exception_on_relative(self, tools):
        rel = Path(ORDERS_FILE).name
        # Must not raise FileNotFoundError (the reported bug).
        result = tools["get_source"](rel, 1, context=10)
        assert isinstance(result["source"], str)

    def test_get_callers_matches_absolute(self, tools):
        symbols = tools["get_file_outline"](ORDERS_FILE)
        line = next(s for s in symbols if s["name"] == "Process")["line"]
        rel = Path(ORDERS_FILE).name
        rel_result = tools["get_callers"](rel, line)
        abs_result = tools["get_callers"](ORDERS_FILE, line)
        assert rel_result, "relative-path get_callers returned empty"
        assert rel_result == abs_result, (
            "relative-path get_callers differs from absolute-path"
        )

    def test_get_callers_no_exception_on_relative(self, tools):
        symbols = tools["get_file_outline"](ORDERS_FILE)
        line = next(s for s in symbols if s["name"] == "Process")["line"]
        rel = Path(ORDERS_FILE).name
        # Must not raise FileNotFoundError (the reported bug).
        result = tools["get_callers"](rel, line)
        assert isinstance(result, list)

    def test_get_callees_matches_absolute(self, tools):
        symbols = tools["get_file_outline"](ORDERS_FILE)
        line = next(s for s in symbols if s["name"] == "Process")["line"]
        rel = Path(ORDERS_FILE).name
        rel_result = tools["get_callees"](rel, line)
        abs_result = tools["get_callees"](ORDERS_FILE, line)
        assert rel_result, "relative-path get_callees returned empty"
        assert rel_result == abs_result, (
            "relative-path get_callees differs from absolute-path"
        )

    def test_get_callees_no_exception_on_relative(self, tools):
        symbols = tools["get_file_outline"](ORDERS_FILE)
        line = next(s for s in symbols if s["name"] == "Process")["line"]
        rel = Path(ORDERS_FILE).name
        # Must not raise FileNotFoundError (the reported bug).
        result = tools["get_callees"](rel, line)
        assert isinstance(result, list)

    def test_relative_result_file_field_is_absolute(self, tools):
        """The `file` field in outline results must be the normalized
        absolute path even when the input was a bare filename, so callers
        (and downstream tools) can pass it back in without re-normalizing."""
        rel = Path(ORDERS_FILE).name
        symbols = tools["get_file_outline"](rel)
        for s in symbols:
            assert s["file"] == ORDERS_FILE, (
                f"outline file field is {s['file']!r}, expected "
                f"{ORDERS_FILE!r} (the normalized absolute path)"
            )


# set_language_server_root restarts a stale gopls
#
# The report: after re-setting the language-server root (even to the same
# root), per-file queries kept failing as if the stale gopls was reused.
# The tracer's _get_lsp assigned the client to self._lsp BEFORE start(); if
# start() raised, self._lsp pointed at a never-started (poisoned) client that
# every later call reused without retrying.  The fix assigns on success only
# and tears down a failed-boot process so no gopls is orphaned.
class TestSetLanguageServerRootRestartsGopls:
    """Re-setting the root — even to the same value — reboots gopls.

    Built on a dedicated GoTracer (not the shared `tools` fixture) so the
    test can inspect ``tracer._lsp`` directly: after a same-root re-set the
    next query must use a NEW, started client, not the previous instance.
    """

    @staticmethod
    def _basename(path: str) -> str:
        return path.replace("\\", "/").rstrip("/").split("/")[-1]

    def _make_tracer(self, root: str):
        from marketplace.plugins.context_tools.extension.cache import CodeCache
        from marketplace.plugins.context_tools.extension.go_tracer import GoTracer

        cache = CodeCache(str(Path(root) / ".metalgate" / "ctx_test.db"))
        return GoTracer(root=root, cache=cache)

    @staticmethod
    def _clear_definitions(tracer) -> None:
        """Clear the definition cache so the next goto_definition re-triggers
        an LSP boot instead of returning a cached result."""
        conn = tracer.cache._conn()
        conn.execute("DELETE FROM definitions")
        conn.commit()

    def test_same_root_reset_starts_fresh_client(self, tools):
        """Re-setting to the SAME root must tear down gopls and boot a new,
        started client on the next query (not reuse the old instance).

        The definition cache is cleared between calls so each goto_definition
        genuinely re-triggers an LSP boot rather than returning a cached
        result, keeping the test independent of future cache changes.
        """
        tracer = self._make_tracer(str(SAMPLE_DIR))
        try:
            # Force gopls to boot by resolving a symbol (needs the LSP).
            tracer.goto_definition(ORDERS_FILE, 40, "o.Process")
            first_lsp = tracer._lsp
            assert first_lsp is not None, "gopls did not start on first query"
            first_id = id(first_lsp)

            # Re-set to the same root — tears down the running gopls.
            tracer.set_root(str(SAMPLE_DIR))
            # Clear the cache so the next call re-enters _get_lsp instead of
            # returning the previously-cached resolution.
            self._clear_definitions(tracer)

            # Next query must boot a fresh client (id changed) and it must
            # be started (reachable), not a poisoned never-started one.
            tracer.goto_definition(ORDERS_FILE, 40, "o.Process")
            second_lsp = tracer._lsp
            assert second_lsp is not None, "gopls did not restart after set_root"
            assert id(second_lsp) != first_id, (
                "set_root reused the previous gopls client instead of booting "
                "a fresh one"
            )
        finally:
            tracer.stop()

    def test_failed_start_does_not_poison_next_call(self, tools):
        """A failed gopls start must leave self._lsp None so the next call
        retries with a fresh client, rather than reusing a never-started
        (poisoned) one permanently for the session."""
        from marketplace.plugins.context_tools.extension.gopls_lsp_client import (
            GoplsLspClient,
        )

        tracer = self._make_tracer(str(SAMPLE_DIR))
        original_start = GoplsLspClient.start
        calls = {"n": 0}

        def flaky_start(self, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("simulated boot failure")
            return original_start(self, *args, **kwargs)

        GoplsLspClient.start = flaky_start
        try:
            # First _get_lsp() raises (simulated boot failure); self._lsp must
            # stay None — the failure is retryable, not permanent.
            with pytest.raises(RuntimeError):
                tracer._get_lsp()
            assert tracer._lsp is None, (
                "self._lsp was assigned despite start() raising — the client "
                "is poisoned and will be reused without retrying"
            )

            # Second call must retry (not reuse the broken client) and succeed.
            lsp = tracer._get_lsp()
            assert lsp is not None, "retry did not produce a client"
            assert tracer._lsp is lsp, "self._lsp not assigned on the retry"
        finally:
            GoplsLspClient.start = original_start
            tracer.stop()
