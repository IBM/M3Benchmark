# Server Backend Architecture

This document describes the server-side architecture of the benchmark — one
section per task, each covering its container, the services running inside it,
the MCP layer the benchmark client connects to, and a full end-to-end diagram.

---

## System Overview

```
                    +------------------------------------------+
                    |          LLM Provider API                |
                    |  OpenAI | Anthropic | Ollama | LiteLLM  |
                    +-------------------+----------------------+
                                        | HTTPS (chat completions)
= = = = = = = = = = = = = = = = = = = = + = = = = = = = HOST = = = = = = = = = = = = =
                                        |
              +─────────────────────────+──────────────────────────+
              |              benchmark_runner.py                   |
              |                                                    |
              |  · Reads domain questions from data/               |
              |  · Runs a ReAct agent loop (LangGraph)             |
              |  · Calls LLM API above for reasoning               |
              |  · Calls MCP tools below for data access           |
              |  · Scores answers → output/{task}/{domain}.json    |
              +─────────────────────────+──────────────────────────+
                                        |
                           docker exec -i, MCP stdio
                           CAPABILITY_ID=N, MCP_DOMAIN=<domain>
                                        |
= = = = = = = = = = = = = = = = = = = = + = = = CONTAINERS = = = = = = = = = = = = = =
                                        |
     +──────────────────────────────────+──────────────────────────────────────+
     |                           image: m3_environ                             |
     |                                                                         |
     |  +──────────────+ +──────────────+ +──────────────+ +─────────────────+ |
     |  | cap_1_bi_apis | | cap_2_dashboard | | cap_3_multihop | | cap_4_multiturn | |
     |  |  Sel / Slot  | |  M3 REST     | |  BPO + M3    | |  M3 REST +      | |
     |  |  MCP server  | |  MCP server  | |  REST router | |  Retriever      | |
     |  +──────┬───────+ +──────┬───────+ +──────┬───────+ +────┬───────┬────+ |
     |         |                |                |              |       |       |
     |         +────────────────+────────────────+--------------+       |       |
     |                                                                  |       |
     |          mcp_dispatch.py  (os.execv → per-task MCP server)      |       |
     |                     |                                            |       |
     |         FastAPI :8000 — M3 REST API (all tasks)    FastAPI :8001 |       |
     |                     |                              Retriever API  |       |
     |                     |                              (capability_4 only)  |       |
     |        +────────────+──────────────────+   +────────────────────+|       |
     |        |  SQLite  /app/db/             |   |  ChromaDB           +       |
     |        |  60+ domain databases         |   |  62 collections             |
     |        |  (capabilities 1, 2, 3, 4)           |   |  (capability_4 only)              |
     |        +───────────────────────────────+   +─────────────────────────────+
     +─────────────────────────────────────────────────────────────────────────+
```

---

## Shared Infrastructure

### One image, four containers

All tasks share a single Docker image (`m3_environ`) built from
[`docker/Dockerfile.unified`](docker/Dockerfile.unified). Four named containers
are started from this image, one per task. The image bundles every server
component; which pieces are active depends on the container's entrypoint and
the `docker exec` command used to start the MCP server.

### How the benchmark connects

The benchmark runner never opens a network socket to a container. Instead it
runs:

```
docker exec -i -e CAPABILITY_ID=<N> -e MCP_DOMAIN=<domain> <container> python /app/mcp_dispatch.py
```

That spawns a short-lived MCP server process inside the container. The process
speaks the [MCP stdio protocol](https://modelcontextprotocol.io) back over
stdin/stdout. The benchmark client (`benchmark/mcp_client.py`) wraps this as
an MCP `ClientSession` and calls `list_tools()` / `call_tool()` as normal.

### MCP dispatcher (`mcp_dispatch.py`)

All tasks share a single container entrypoint — `/app/mcp_dispatch.py`. It
reads the `CAPABILITY_ID` environment variable and calls `os.execv()` to replace
itself with the correct server process. There is no proxy layer; the stdio
pipe connects directly to the target server after exec.

| `CAPABILITY_ID` | Exec target |
|-----------|-------------|
| `1` | `python -m apis.m3.python_tools.mcp` |
| `2` | `python /app/m3-rest/mcp_server.py` |
| `3` | `python /app/environment/bpo/mcp/bpo_router.py` |
| `4` | `python /app/retrievers/capability_4_mcp_server.py` |

The original server scripts remain intact and are still tested directly by
`docker/smoke_test.sh`. The dispatcher is just a thin routing shim.

### Container startup (`entrypoint.sh`)

When a container first starts,
[`docker/entrypoint-unified.sh`](docker/entrypoint-unified.sh) launches the
long-lived FastAPI background services and waits for them to become healthy
before declaring the container ready:

| Port | Service | Health check |
|------|---------|--------------|
| 8000 | M3 REST FastAPI | `GET /openapi.json` |
| 8001 | Retriever FastAPI | `GET /health` (skipped if no `chroma_data/`) |

Both services stay up indefinitely and handle all subsequent `docker exec`
invocations for that container's lifetime.

---

## Task 1 — Slot-Filling / Selection (RouterMCPServer)

### Purpose

The agent must identify the correct tool and fill its parameter slots from a
natural-language query. A single MCP entry point exposes either a generic
slot-filling toolset or a specialised selection toolset (with dynamically
generated column getters), switching between them at runtime based on which
"tool universe" the agent selects for a given query.

### Container: `capability_1_bi_apis_m3_environ`

| What | Detail |
|------|--------|
| FastAPI services | M3 REST on port 8000 (started but not used by MCP) |
| MCP entry point | `python -m apis.m3.python_tools.mcp` |
| Entry CLI | [`apis/m3/python_tools/mcp/cli.py`](apis/m3/python_tools/mcp/cli.py) |
| Server factory | [`apis/m3/python_tools/mcp/mcp_server.py`](apis/m3/python_tools/mcp/mcp_server.py) — `create_server()` |
| Active server class | `RouterMCPServer` (selected via `MCP_SERVER_TYPE=router`) |

### How it works

`cli.py` reads environment variables and calls `create_server()`, which
instantiates a **`RouterMCPServer`** (the default when `MCP_SERVER_TYPE=router`).

At startup the router:

1. Reads `mcp_tool_universe_id_mapping.yaml` and filters the universe list to
   entries whose `domain` field matches `MCP_DOMAIN`.
2. Initialises **both** `SlotFillingTools` and `SelectionTools` in memory.
3. Loads the first universe's data from `SQLite` (`/app/db/{domain}/{domain}.sqlite`)
   into `active_data`.
4. Defaults to the **slot-filling** toolset as the active tool set.

The agent interacts with the server through two mechanisms:

- **`list_tools()`** — returns whichever toolset is currently active.
- **`call_tool("get_data", tool_universe_id=<uid>)`** — switches `active_data`
  to the requested universe; if the universe's YAML entry contains
  `server_type: selection`, the router also switches the active toolset to
  `SelectionTools` (regenerating column-level getter functions from the new
  schema). A subsequent `list_tools()` then returns the selection toolset.

No HTTP calls are made during tool execution — all data access is direct Python
reads from the pre-loaded SQLite database.

### Tool sets

| Toolset | Active when | Tools exposed |
|---------|-------------|---------------|
| **Slot-filling** | universe `server_type` absent or `slot_filling` | `filter_data`, `sort_data`, `aggregate_data`, `transform_data`, `retrieve_data`, `Calculator`, `concatenate_data`, `select_unique_values`, `peek_fcn`, `get_data` |
| **Selection** | universe `server_type: selection` | 8 × `select_data_*` filters, 2 × `sort_data_*`, 8 × `compute_data_*` aggregates, `truncate`, 3 × `transform_data_to_*`, `Calculator`, `concatenate_data`, `select_unique_values`, dynamically generated `get_{column}` getters, `get_data` |

### Stack diagram

```
Benchmark Client (host)
        │
        │  docker exec -i
        │    -e CAPABILITY_ID=1
        │    -e MCP_DOMAIN=<domain>
        │    -e MCP_SERVER_TYPE=router
        │    -e MCP_DB_ROOT=/app/db
        │  capability_1_bi_apis_m3_environ
        │  python /app/mcp_dispatch.py
─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ host / container boundary ─ ─ ─ ─
        ▼
┌────────────────────────────────────────────────────────────┐
│  capability_1_bi_apis_m3_environ                                         │
│                                                            │
│  mcp_dispatch.py  (CAPABILITY_ID=1)                              │
│    └─ os.execv → python -m apis.m3.python_tools.mcp        │
│                                                            │
│  cli.py → create_server() → RouterMCPServer  (stdio)       │
│                                                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  RouterMCPServer                                    │   │
│  │                                                     │   │
│  │  startup:                                           │   │
│  │    • load mcp_tool_universe_id_mapping.yaml         │   │
│  │    • filter universes to MCP_DOMAIN                 │   │
│  │    • init SlotFillingTools + SelectionTools         │   │
│  │    • load active_data from SQLite (first universe)  │   │
│  │    • active toolset = slot_filling                  │   │
│  │                                                     │   │
│  │  list_tools()                                       │   │
│  │    → current active toolset tools                   │   │
│  │                                                     │   │
│  │  call_tool("get_data", tool_universe_id=<uid>)      │   │
│  │    → reload active_data from SQLite                 │   │
│  │    → if universe server_type == "selection":        │   │
│  │        regenerate get_{col} getters from schema     │   │
│  │        switch active toolset → SelectionTools       │   │
│  │    → else: active toolset → SlotFillingTools        │   │
│  │                                                     │   │
│  │  call_tool(<other>, ...)                            │   │
│  │    → direct Python call on active toolset           │   │
│  └─────────────────────┬───────────────────────────────┘   │
│                        │                                   │
│            ┌───────────┴──────────────┐                    │
│            ▼                          ▼                    │
│  mcp_tool_universe_id_mapping.yaml    SQLite               │
│  /app/apis/configs/                   /app/db/{domain}/    │
│  (universe IDs, init_args,            {domain}.sqlite      │
│   server_type per query)                                   │
└────────────────────────────────────────────────────────────┘
```

---

## Task 2 — M3 REST SQL Tools

### Purpose

The agent answers questions by calling SQL-backed REST endpoints for a single
domain. Tools are auto-discovered from the FastAPI OpenAPI spec so the tool
list reflects the live schema without any manual registration.

### Container: `capability_2_dashboard_apis_m3_environ`

| What | Detail |
|------|--------|
| FastAPI services | M3 REST on port 8000 |
| MCP entry point | `python /app/m3-rest/mcp_server.py` |
| MCP source | [`apis/m3/rest/mcp_server.py`](apis/m3/rest/mcp_server.py) |

### How it works

`FastAPIMCPServer` fetches `/openapi.json` from the M3 REST FastAPI on startup,
filters routes to those under `/v1/{MCP_DOMAIN}/`, and converts each route to
an MCP `Tool`. On `call_tool`, it reconstructs the HTTP request (path params,
query params, request body) and proxies it to FastAPI, which executes the SQL
query against the domain's SQLite database.

### Stack diagram

```
Benchmark Client (host)
        │
        │  docker exec -i -e CAPABILITY_ID=2 -e MCP_DOMAIN=address
        │  capability_2_dashboard_apis_m3_environ
        │  python /app/mcp_dispatch.py
─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ host / container boundary ─ ─ ─ ─
        ▼
┌──────────────────────────────────────────────────┐
│  capability_2_dashboard_apis_m3_environ                               │
│                                                  │
│  mcp_dispatch.py  (CAPABILITY_ID=2)                    │
│    └─ os.execv → python /app/m3-rest/mcp_server.py│
│                                                  │
│  ┌────────────────────────────────────────────┐  │
│  │  FastAPIMCPServer  (stdio)                 │  │
│  │                                            │  │
│  │  • list_tools()  →  GET :8000/openapi.json │  │
│  │  • call_tool()   →  HTTP → FastAPI         │  │
│  └──────────────────┬─────────────────────────┘  │
│                     │  HTTP :8000                │
│                     ▼                            │
│         M3 REST FastAPI  (uvicorn, port 8000)    │
│         /v1/{domain}/* routes (≈40 per domain)   │
│                     │                            │
│                     ▼                            │
│         SQLite  /app/db/{domain}/*.sqlite        │
└──────────────────────────────────────────────────┘
```

### M3 REST FastAPI — route shape

```
GET  /v1/{domain}/{resource}              → SQL SELECT all
GET  /v1/{domain}/{resource}/{id}         → SQL SELECT by PK
POST /v1/{domain}/{resource}/filter       → SQL WHERE clause
...  (~40 routes per domain, 45+ domains)
```

Source: [`apis/m3/rest/app.py`](apis/m3/rest/app.py), per-domain routers in
[`apis/m3/rest/server/`](apis/m3/rest/server/).

---

## Task 3 — BPO / M3 REST Router

### Purpose

The benchmark tests routing across two heterogeneous tool sets: the **BPO**
(Business Process Outsourcing) tools for the `bpo` domain, and the standard
M3 REST SQL tools for all other domains. A single MCP entry point handles
both by exec-replacing itself with the correct server at startup.

### Container: `capability_3_multihop_reasoning_m3_environ`

| What | Detail |
|------|--------|
| FastAPI services | M3 REST on port 8000 |
| MCP entry point | `python /app/environment/bpo/mcp/bpo_router.py` |
| Router source | [`environment/bpo/mcp/bpo_router.py`](environment/bpo/mcp/bpo_router.py) |
| BPO server source | [`apis/bpo/mcp/server.py`](apis/bpo/mcp/server.py) |
| M3 server source | [`apis/m3/rest/mcp_server.py`](apis/m3/rest/mcp_server.py) |

### How it works

`bpo_router.py` is a six-line shim. It reads `MCP_DOMAIN` and calls
`os.execv` to **replace itself** with the target server process — there is no
proxy layer. The MCP client's stdio pipe connects directly to the chosen server.

```python
target = BPO_SERVER if domain == "bpo" else M3_REST_SERVER
os.execv(sys.executable, [sys.executable, target])
```

- **BPO server** (`bpo` domain): uses the `FastMCP` framework with
  `@mcp.tool()` decorators. Tools are statically defined and call in-process
  Python functions against BPO data — no FastAPI involved.
- **M3 REST server** (all other domains): same `FastAPIMCPServer` as Task 2,
  wrapping the M3 REST FastAPI on port 8000.

### Stack diagram

```
Benchmark Client (host)
        │
        │  docker exec -i -e CAPABILITY_ID=3 -e MCP_DOMAIN=bpo   (or MCP_DOMAIN=airline)
        │  capability_3_multihop_reasoning_m3_environ
        │  python /app/mcp_dispatch.py
─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ host / container boundary ─ ─ ─ ─
        ▼
┌──────────────────────────────────────────────────────────────┐
│  capability_3_multihop_reasoning_m3_environ                                           │
│                                                              │
│  mcp_dispatch.py  (CAPABILITY_ID=3)                                │
│    └─ os.execv → bpo_router.py  ──┬── MCP_DOMAIN=bpo ───► │
│                                   │                          │
│                        ┌──────────┴──────────────────────┐  │
│                        │  BPO FastMCP server  (stdio)    │  │
│                        │  @mcp.tool() — 30+ static tools │  │
│                        │  → in-process BPO Python APIs   │  │
│                        └─────────────────────────────────┘  │
│                                   │                          │
│                                   └── MCP_DOMAIN=<other> ►  │
│                                                              │
│                        ┌─────────────────────────────────┐  │
│                        │  FastAPIMCPServer  (stdio)       │  │
│                        │  → GET :8000/openapi.json        │  │
│                        │  → HTTP /v1/{domain}/...         │  │
│                        └──────────────┬──────────────────┘  │
│                                       │  HTTP :8000         │
│                                       ▼                     │
│                         M3 REST FastAPI  (port 8000)        │
│                                       │                     │
│                                       ▼                     │
│                         SQLite  /app/db/{domain}/*.sqlite   │
└──────────────────────────────────────────────────────────────┘
```

---

## Task 5 — Combined M3 REST + Retriever

### Purpose

The agent has access to both structured SQL tools (M3 REST) and unstructured
semantic search (ChromaDB retriever) in a single session. Tool filtering is
**asymmetric by design**:

- **M3 REST tools** are scoped to the primary domain only (e.g. `/v1/address/*`).
- **Retriever tools** are exposed for the primary domain **plus its negative
  (distractor) domains** looked up from `domain_negatives.json`. For `address`
  this means `query_address`, `query_olympics`, `query_card_games`,
  `query_legislator`, and `query_craftbeer`. The agent must retrieve from the
  correct collection despite having access to confusable alternatives.

### Container: `capability_4_multiturn_m3_environ`

| What | Detail |
|------|--------|
| FastAPI services | M3 REST on port 8000 + Retriever on port 8001 |
| MCP entry point | `python /app/retrievers/capability_4_mcp_server.py` |
| MCP source | [`environment/retrievers/capability_4_mcp_server.py`](environment/retrievers/capability_4_mcp_server.py) |
| Negatives config | [`apis/retrievers/domain_negatives.json`](apis/retrievers/domain_negatives.json) |

### How it works

`Capability4CombinedMCPServer` fetches `/openapi.json` from **both** FastAPI servers
at startup, builds a merged `tools_cache`, and stores the originating
`backend_url` in each tool's `_metadata`. On `call_tool` it looks up the tool,
reads `_metadata["backend_url"]`, and routes the HTTP request to the correct
service.

The negatives for the primary domain are resolved once in `main()` via
`load_retriever_domains()`, which reads `domain_negatives.json` and expands the
primary domain list before the server is constructed.

| Backend | Filter | Tools exposed | Tool name pattern |
|---------|--------|--------------|-------------------|
| M3 REST `:8000` | primary domain only | ~40 | operationId from OpenAPI (e.g. `get_address_streets`) |
| Retriever `:8001` | primary + negatives | ~5 | `query_{domain}` (e.g. `query_address`, `query_olympics`) |

### Stack diagram

```
Benchmark Client (host)
        │
        │  docker exec -i -e CAPABILITY_ID=5 -e MCP_DOMAIN=address
        │  capability_4_multiturn_m3_environ
        │  python /app/mcp_dispatch.py
─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ host / container boundary ─ ─ ─ ─
        ▼
┌────────────────────────────────────────────────────────────────────┐
│  capability_4_multiturn_m3_environ                                                 │
│                                                                    │
│  mcp_dispatch.py  (CAPABILITY_ID=5)                                      │
│    └─ os.execv → python /app/retrievers/capability_4_mcp_server.py │
│                                                                    │
│  domain_negatives.json["address"]                                  │
│    → [address, olympics, card_games, legislator, craftbeer]        │
│                          │                                         │
│  ┌───────────────────────▼────────────────────────────────────┐   │
│  │  Capability4CombinedMCPServer  (stdio)                     │   │
│  │                                                            │   │
│  │  list_tools()                                              │   │
│  │    ├─ GET :8000/openapi.json                               │   │
│  │    │    filter: /v1/address/* → M3 REST tools (≈40)       │   │
│  │    └─ GET :8001/openapi.json                               │   │
│  │         filter: address + negatives → query_address,       │   │
│  │                  query_olympics, query_card_games, ...  (5) │   │
│  │                                                            │   │
│  │  call_tool(name, args)   [routes via _metadata.backend_url]│   │
│  │    ├─ M3 REST tool  → HTTP → :8000/v1/address/...         │   │
│  │    └─ query_*       → HTTP → :8001/{domain}/query         │   │
│  └────────────┬──────────────────────────┬──────────────────┘    │
│               │  HTTP :8000              │  HTTP :8001            │
│               ▼                          ▼                        │
│   M3 REST FastAPI                 Retriever FastAPI               │
│   (port 8000)                     (port 8001)                     │
│               │                          │                        │
│               ▼                          ▼                        │
│   SQLite                          ChromaDB collections            │
│   /app/db/{domain}/               (one per domain)                │
│   *.sqlite                        /app/retrievers/chroma_data/    │
└────────────────────────────────────────────────────────────────────┘
```

### Retriever FastAPI — route shape

```
POST /{domain}/query   body: { query: str, n_results: int }
                       → ranked document chunks (ChromaDB semantic search)
```

Source: [`apis/retrievers/server.py`](apis/retrievers/server.py). Embeddings
model: `ibm-granite/granite-embedding-english-r2` (pre-downloaded into the
image; takes up to 5 min to warm up on first container start).

---

## Key Source Files

| File | Role |
|------|------|
| [`docker/Dockerfile.unified`](docker/Dockerfile.unified) | Single image for all tasks |
| [`docker/entrypoint-unified.sh`](docker/entrypoint-unified.sh) | Starts FastAPI services; waits for health |
| [`docker/mcp_dispatch.py`](docker/mcp_dispatch.py) | Single MCP entrypoint — reads `CAPABILITY_ID`, `os.execv()`s into the right server |
| [`benchmark/mcp_connection_config.yaml`](benchmark/mcp_connection_config.yaml) | Maps task IDs → containers + MCP commands |
| [`benchmark/mcp_client.py`](benchmark/mcp_client.py) | Builds `docker exec` command; opens MCP `ClientSession` |
| [`apis/m3/rest/app.py`](apis/m3/rest/app.py) | M3 REST FastAPI — 45+ domain routers, port 8000 |
| [`apis/m3/rest/mcp_server.py`](apis/m3/rest/mcp_server.py) | OpenAPI→MCP wrapper for M3 REST (Tasks 2 & 3) |
| [`apis/m3/python_tools/mcp/cli.py`](apis/m3/python_tools/mcp/cli.py) | CLI entry point; reads env vars, calls `create_server()` (Task 1) |
| [`apis/m3/python_tools/mcp/mcp_server.py`](apis/m3/python_tools/mcp/mcp_server.py) | `RouterMCPServer` / `SlotFillingMCPServer` / `SelectionMCPServer` + `create_server()` factory (Task 1) |
| [`apis/m3/python_tools/mcp/config.py`](apis/m3/python_tools/mcp/config.py) | `MCPServerConfig` dataclass; resolves `MCP_DOMAIN` → SQLite path (Task 1) |
| [`apis/configs/mcp_tool_universe_id_mapping.yaml`](apis/configs/mcp_tool_universe_id_mapping.yaml) | Tool universe registry — universe IDs, init args, `server_type` per query (Task 1) |
| [`apis/bpo/mcp/server.py`](apis/bpo/mcp/server.py) | BPO FastMCP server — `@mcp.tool()` decorators (Task 3) |
| [`environment/bpo/mcp/bpo_router.py`](environment/bpo/mcp/bpo_router.py) | `os.execv` router → BPO or M3 REST (Task 3) |
| [`apis/retrievers/server.py`](apis/retrievers/server.py) | Retriever FastAPI — ChromaDB, port 8001 |
| [`environment/retrievers/capability_4_mcp_server.py`](environment/retrievers/capability_4_mcp_server.py) | Combined MCP server — merges M3 REST + Retriever (Capability 4) |
| [`apis/retrievers/domain_negatives.json`](apis/retrievers/domain_negatives.json) | Maps each domain to its distractor domains for retriever tool expansion (Task 5) |
