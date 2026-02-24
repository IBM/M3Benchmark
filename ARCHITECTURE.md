# Server Backend Architecture

This document describes the server-side architecture of the benchmark — one
section per task, each covering its container, the services running inside it,
the MCP layer the benchmark client connects to, and a full end-to-end diagram.

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
docker exec -i -e MCP_DOMAIN=<domain> <container> python <mcp_server.py>
```

That spawns a short-lived MCP server process inside the container. The process
speaks the [MCP stdio protocol](https://modelcontextprotocol.io) back over
stdin/stdout. The benchmark client (`benchmark/mcp_client.py`) wraps this as
an MCP `ClientSession` and calls `list_tools()` / `call_tool()` as normal.

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

## Task 1 — Slot-Filling (Sel/Slot MCP Server)

### Purpose

The agent must identify the correct tool and fill its parameter slots from a
natural-language query.  Tools are loaded from a YAML "universe" configuration
scoped to the requested domain.

### Container: `task_1_m3_environ`

| What | Detail |
|------|--------|
| FastAPI services | M3 REST on port 8000 (started but not used by MCP) |
| MCP entry point | `python -m apis.m3.python_tools.mcp` |
| MCP source | [`apis/m3/python_tools/mcp/mcp_server.py`](apis/m3/python_tools/mcp/mcp_server.py) |

### How it works

`SlotFillingMCPServer` is a **declarative** MCP server — it does not wrap the
M3 REST FastAPI. Instead it loads a YAML universe configuration that defines
which filter / sort / aggregate / retrieval tools are available for the domain,
and reads domain data directly in Python. No HTTP calls are made during tool
execution.

### Stack diagram

```
Benchmark Client (host)
        │
        │  docker exec -i -e MCP_DOMAIN=<domain>
        │  task_1_m3_environ
        │  python -m apis.m3.python_tools.mcp
        │
        ▼
┌──────────────────────────────────────────────────┐
│  task_1_m3_environ                               │
│                                                  │
│  ┌────────────────────────────────────────────┐  │
│  │  SlotFillingMCPServer  (stdio)             │  │
│  │                                            │  │
│  │  • list_tools()  →  reads YAML universe    │  │
│  │  • call_tool()   →  direct Python call     │  │
│  └──────────────────┬─────────────────────────┘  │
│                     │                            │
│                     ▼                            │
│         tool universe YAML + domain data         │
│         /app/apis/configs/                       │
└──────────────────────────────────────────────────┘
```

---

## Task 2 — M3 REST SQL Tools

### Purpose

The agent answers questions by calling SQL-backed REST endpoints for a single
domain. Tools are auto-discovered from the FastAPI OpenAPI spec so the tool
list reflects the live schema without any manual registration.

### Container: `task_2_m3_environ`

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
        │  docker exec -i -e MCP_DOMAIN=address
        │  task_2_m3_environ
        │  python /app/m3-rest/mcp_server.py
        │
        ▼
┌──────────────────────────────────────────────────┐
│  task_2_m3_environ                               │
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

### Container: `task_3_m3_environ`

| What | Detail |
|------|--------|
| FastAPI services | M3 REST on port 8000 |
| MCP entry point | `python /app/apis/bpo/mcp/task3_router.py` |
| Router source | [`apis/bpo/mcp/task3_router.py`](apis/bpo/mcp/task3_router.py) |
| BPO server source | [`apis/bpo/mcp/server.py`](apis/bpo/mcp/server.py) |
| M3 server source | [`apis/m3/rest/mcp_server.py`](apis/m3/rest/mcp_server.py) |

### How it works

`task3_router.py` is a six-line shim. It reads `MCP_DOMAIN` and calls
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
        │  docker exec -i -e MCP_DOMAIN=bpo      (or -e MCP_DOMAIN=airline)
        │  task_3_m3_environ
        │  python /app/apis/bpo/mcp/task3_router.py
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│  task_3_m3_environ                                           │
│                                                              │
│  task3_router.py  ──  os.execv  ──┬── MCP_DOMAIN=bpo ────►  │
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

### Container: `task_5_m3_environ`

| What | Detail |
|------|--------|
| FastAPI services | M3 REST on port 8000 + Retriever on port 8001 |
| MCP entry point | `python /app/retrievers/task5_mcp_server.py` |
| MCP source | [`apis/retrievers/task5_mcp_server.py`](apis/retrievers/task5_mcp_server.py) |
| Negatives config | [`apis/retrievers/domain_negatives.json`](apis/retrievers/domain_negatives.json) |

### How it works

`Task5CombinedMCPServer` fetches `/openapi.json` from **both** FastAPI servers
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
        │  docker exec -i -e MCP_DOMAIN=address
        │  task_5_m3_environ
        │  python /app/retrievers/task5_mcp_server.py
        │
        ▼
┌────────────────────────────────────────────────────────────────────┐
│  task_5_m3_environ                                                 │
│                                                                    │
│  domain_negatives.json["address"]                                  │
│    → [address, olympics, card_games, legislator, craftbeer]        │
│                          │                                         │
│  ┌───────────────────────▼────────────────────────────────────┐   │
│  │  Task5CombinedMCPServer  (stdio)                           │   │
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
| [`benchmark/mcp_connection_config.yaml`](benchmark/mcp_connection_config.yaml) | Maps task IDs → containers + MCP commands |
| [`benchmark/mcp_client.py`](benchmark/mcp_client.py) | Builds `docker exec` command; opens MCP `ClientSession` |
| [`apis/m3/rest/app.py`](apis/m3/rest/app.py) | M3 REST FastAPI — 45+ domain routers, port 8000 |
| [`apis/m3/rest/mcp_server.py`](apis/m3/rest/mcp_server.py) | OpenAPI→MCP wrapper for M3 REST (Tasks 2 & 3) |
| [`apis/m3/python_tools/mcp/mcp_server.py`](apis/m3/python_tools/mcp/mcp_server.py) | Declarative Sel/Slot MCP server (Task 1) |
| [`apis/bpo/mcp/server.py`](apis/bpo/mcp/server.py) | BPO FastMCP server — `@mcp.tool()` decorators (Task 3) |
| [`apis/bpo/mcp/task3_router.py`](apis/bpo/mcp/task3_router.py) | `os.execv` router → BPO or M3 REST (Task 3) |
| [`apis/retrievers/server.py`](apis/retrievers/server.py) | Retriever FastAPI — ChromaDB, port 8001 |
| [`apis/retrievers/task5_mcp_server.py`](apis/retrievers/task5_mcp_server.py) | Combined MCP server — merges M3 REST + Retriever (Task 5) |
| [`apis/retrievers/domain_negatives.json`](apis/retrievers/domain_negatives.json) | Maps each domain to its distractor domains for retriever tool expansion (Task 5) |
