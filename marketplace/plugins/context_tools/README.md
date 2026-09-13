# Context Tools

Code-navigation tools (goto definition, outline, source, callers, callees,
find symbol) backed by a language server (gopls for Go, ty for Python) and
tree-sitter.

## Prerequisites (Go)

The Go per-file tools resolve symbols through `gopls`, which builds a
**workspace view** from the nearest `go.work` file.  That view must include
the module you are navigating.

- The target Go module must be listed in the nearest `go.work`'s `use` block.
  If there is no `go.work` above the module, gopls builds a single-module view
  from the module's own `go.mod` — which is usually what you want.
- Run `go mod download` so the dependency graph is populated before gopls
  starts; gopls needs the module graph resolved to index the workspace.

### Symptom of a misconfigured root

Per-file tools (`get_file_outline`, `goto_definition`, `get_callers`,
`get_callees`) return empty while `find_symbol` still returns results: the
module is indexed for symbol search but not included in the per-file view.

### Fix

1. Add the module to the nearest `go.work` `use` block (or remove a stray
   `go.work` so gopls falls back to the single-module view).
2. Re-set the language-server root via `set_language_server_root` (even to
   the same root) to tear down and reboot gopls so it re-reads the updated
   `go.work` and rebuilds the view.
