# deepagents-code-cli

A uv project that launches the [deepagents code](https://docs.langchain.com/oss/deepagents/code/overview) (`dcode`) CLI with:

- **`evroc`** — a pip-installed model provider package (`ChatOpenAI` subclass) wired through `config.toml` as an arbitrary provider via `class_path`. Models are fetched dynamically from the evroc API at launch and written to `evroc/data/_profiles.py` so the `/model` switcher is populated.
- **`dynamic_tools`** — a dcode marketplace plugin that dynamically loads `.py` tool files from `.metalgate/tools` (scanned under the current project root **and** the agent project root) and registers their callables as model tools. Installed automatically by `run.sh`.

## How it works

### Model provider (`evroc` package)

The evroc endpoint (`https://models.think.cloud.evroc.com/v1`) speaks the OpenAI Chat Completions API. `evroc` is a pip-installable package providing `ChatModel` — a thin `ChatOpenAI` subclass. `config.toml` registers it as an arbitrary provider:

```toml
[models.providers.evroc]
class_path = "evroc:ChatModel"
api_key_env = "EVROC_API_KEY"
base_url = "https://models.think.cloud.evroc.com/v1"
models = ["zai-org/GLM-5.2", ...]

[models.providers.evroc.params]
use_responses_api = false
```

dcode imports `ChatModel` via `importlib.import_module("evroc")` and instantiates it with `model=`, `base_url=`, `api_key=`, and any `params` from the config table.

### Model discovery (`_profiles.py`)

dcode discovers models for `class_path` providers by reading `<package>.data._profiles` — a `_PROFILES` dict bundled inside the package, the same way `langchain_openai/data/_profiles.py` works for the built-in `openai` provider. At launch, `run.sh` calls `evroc.generate_profiles()` which fetches `GET /models` from the evroc API and writes `evroc/data/_profiles.py` dynamically.

### Dynamic tools (marketplace plugin)

Packaged as a dcode plugin inside a [marketplace](https://docs.langchain.com/oss/deepagents/code/plugins#create-a-marketplace). `run.sh` registers the marketplace and installs the plugin; dcode auto-discovers the extension. At startup it scans `TOOLS_DIRNAME` (`.metalgate/tools`) under two roots, de-duplicated by path:

1. **Current project root** (`api.cwd`) — the project the agent is editing. Agent-authored tool files (written by `write_tool_file`) land here.
2. **Agent project root** — derived from the parent of `DEEPAGENTS_HOME` (`run.sh` sets `DEEPAGENTS_HOME=<repo>/.metalgate`, so the parent is the repo root). Project-wide tools defined here are available in every session. Falls back to the current project root when `DEEPAGENTS_HOME` is unset.

Each `.py` file is imported once; every top-level callable (or name in `__all__`) becomes a model tool via `register_tool()`. The extension also exposes three agent-callable tools: `write_tool_file`, `reload_dynamic_tools`, and `list_dynamic_tools`.

> **Caveat:** the first registration for a tool name wins, so editing an already-loaded tool's code does not hot-swap it live — only brand-new function names are picked up without a `/restart`.

> Python extensions require `DEEPAGENTS_CODE_EXPERIMENTAL=1` (set by `run.sh`).

## Project structure

```
pyproject.toml                              # uv project: deepagents-code==0.1.65 + evroc (path dep)
run.sh                                      # fetches models, generates config.toml, installs plugin, launches dcode
evroc/                                      # pip-installable provider package
├── pyproject.toml                          # hatchling build, deps: langchain-openai, requests
└── evroc/
    ├── __init__.py                         # exports ChatModel, fetch_models, generate_profiles
    ├── chat_model.py                       # ChatModel(ChatOpenAI) — class_path = "evroc:ChatModel"
    ├── models.py                           # fetch_models() + generate_profiles() — dynamic discovery
    └── data/
        ├── __init__.py
        └── _profiles.py                    # _PROFILES dict — read by dcode, generated at launch
marketplace/
├── .claude-plugin/marketplace.json         # marketplace catalog (name + plugins[])
└── plugins/dynamic_tools/
    ├── .claude-plugin/plugin.json
    └── extension/
        ├── __init__.py
        ├── dynamic_tools.py                # scan/import/register .py tool files
        └── factory.py                      # wires scanner + agent-facing tools into ExtensionAPI
```

## Setup

```bash
uv python install 3.13
uv sync
```

## Usage

```bash
# Set the required API key, then launch
EVROC_API_KEY=sk-... ./run.sh

# Pass dcode arguments through
EVROC_API_KEY=sk-... ./run.sh --model evroc:zai-org/GLM-5.2
```

### Environment variables

| Variable          | Required | Default                                   | Description                     |
| ----------------- | -------- | ----------------------------------------- | ------------------------------- |
| `EVROC_API_KEY`   | yes      | —                                         | API key for the evroc platform. |
| `EVROC_MODEL`     | no       | `zai-org/GLM-5.2`                         | Default model identifier.       |
| `EVROC_BASE_URL`  | no       | `https://models.think.cloud.evroc.com/v1` | Override the evroc base URL.    |

## Publishing the marketplace

Push `marketplace/` to a Git repository, then users add it:

```bash
dcode plugin marketplace add owner/marketplace-repo
dcode plugin install dynamic_tools@evroc-extensions
```
``
