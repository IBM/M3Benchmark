# Architecture

```
                        +--------------------------+
                        |     Benchmark Runner     |
                        |   (benchmark_runner.py)  |
                        |                          |
                        |  - Loads domain tasks    |
                        |  - Spawns MCP servers    |
                        |  - Runs LLM agent        |
                        |  - Collects results      |
                        +----+----------------+----+
                             |                |
                   MCP stdio |                | MCP stdio
                             |                |
              +--------------+--+          +--+--------------+
              |  Docker Service |          |  Docker Service |
              |  Port 8000      |          |  Port 8001      |
              |                 |          |                 |
         +----+----+-------+   |     +----+----+-------+   |
         |         |       |   |     |         |       |   |
         |   MCP   | FastAPI|   |     |   MCP   | FastAPI|   |
         | Server  | Server |   |     | Server  | Server |   |
         |         |       |   |     |         |       |   |
         +----+----+---+---+   |     +----+----+---+---+   |
              |        |       |          |        |       |
              +--------+       |          +--------+       |
              |                |          |                |
   +----------+----------+    |   +------+------+         |
   |  M3 Tools Service   |    |   |  Retrievers  |         |
   |  (fastapi-mcp)      |    |   |  Service     |         |
   +-----------+----------+    |   +------+-------+         |
               |               |          |                |
               |               |          |                |
+--------------+-----------+   |   +------+--------+       |
|    60+ SQLite Databases  |   |   | ChromaDB       |       |
|    /db/{domain}/*.sqlite |   |   | (vector store) |       |
+--------------------------+   |   +----------------+       |
                               |                            |
                               +----------------------------+
```

## Two Docker Services

### 1. M3 Tools Service (port 8000)

Database query tools for 60+ domains (hockey, airline, financial, etc.)

```
+-----------------------------------------------------------+
|                  Container: fastapi-mcp-server             |
|                                                           |
|  +-------------------+       +-------------------------+  |
|  |    MCP Server     | stdio  |     FastAPI Server      |  |
|  |  (mcp_server.py)  |<----->|       (app.py)          |  |
|  |                   |       |                         |  |
|  | - Fetches OpenAPI |  HTTP | - 60+ domain routers   |  |
|  |   spec on start   | :8000 | - GET endpoints for     |  |
|  | - Converts to     |       |   database queries      |  |
|  |   MCP tools       |       | - Prometheus metrics    |  |
|  | - MCP_DOMAIN env  |       |                         |  |
|  |   filters tools   |       |                         |  |
|  +-------------------+       +-----------+-------------+  |
|                                          |                |
|                              +-----------+-----------+    |
|                              |   SQLite Databases    |    |
|                              |  /db/hockey.sqlite    |    |
|                              |  /db/airline.sqlite   |    |
|                              |  /db/financial.sqlite |    |
|                              |  /db/...              |    |
|                              +-----------------------+    |
+-----------------------------------------------------------+
```

**Tools exposed:** Domain-specific SQL query wrappers
- `get_goalies_by_minutes_played(min_minutes)`
- `flight_count_by_date(date)`
- `get_movie_rating(title)` ...etc.

---

### 2. Retrievers Service (port 8001)

Semantic search (RAG) over 62 domain document collections.

```
+-----------------------------------------------------------+
|                Container: retriever-mcp-server             |
|                                                           |
|  +-------------------+       +-------------------------+  |
|  |    MCP Server     | stdio  |     FastAPI Server      |  |
|  |  (mcp_server.py)  |<----->|      (server.py)        |  |
|  |                   |       |                         |  |
|  | - Fetches OpenAPI |  HTTP | - /{domain}/query       |  |
|  |   spec on start   | :8001 | - /{domain}/chunks      |  |
|  | - Resolves $ref   |       | - /domains              |  |
|  |   in schemas      |       | - /health               |  |
|  | - Creates per-    |       |                         |  |
|  |   domain tools    |       | - Granite embeddings    |  |
|  |   (query_address, |       |   (ibm-granite/granite- |  |
|  |    query_hockey)  |       |    embedding-english-r2)|  |
|  +-------------------+       +-----------+-------------+  |
|                                          |                |
|                              +-----------+-----------+    |
|                              |      ChromaDB         |    |
|                              |  (PersistentClient)   |    |
|                              |                       |    |
|                              |  Collections:         |    |
|                              |   address (12k docs)  |    |
|                              |   hockey  (8k docs)   |    |
|                              |   ...62 domains       |    |
|                              +-----------------------+    |
+-----------------------------------------------------------+
```

**Tools exposed:** Semantic search per domain
- `query_address(question, n_results)` -> ranked document chunks
- `query_hockey(question, n_results)` -> ranked document chunks ...etc.

---

## Data Flow

```
User Question
      |
      v
+------------------+
| Benchmark Runner |
| (LLM Agent)      |---+
+------------------+   |
                       |  "Which hockey player was born in 1990?"
                       |
         +-------------+-------------+
         |                           |
         v                           v
  +--------------+           +---------------+
  | M3 Tools MCP |           | Retriever MCP |
  | (stdio)      |           | (stdio)       |
  +--------------+           +---------------+
         |                           |
         v                           v
  +--------------+           +---------------+
  | FastAPI :8000|           | FastAPI :8001  |
  +--------------+           +---------------+
         |                           |
         v                           v
  +--------------+           +---------------+
  | SQLite DB    |           | ChromaDB      |
  | (exact query)|           | (vector search)|
  +--------------+           +---------------+
         |                           |
         v                           v
  Structured rows              Ranked documents
  (name, birth_year,           (text chunks with
   country, ...)               similarity scores)
```

## Shared MCP Pattern

Both services follow the same architecture:

1. **FastAPI** serves domain-specific REST endpoints
2. **MCP Server** wraps FastAPI by:
   - Fetching `/openapi.json` at startup
   - Converting endpoints to MCP tools
   - Filtering by `MCP_DOMAIN` env var
   - Proxying tool calls as HTTP requests
3. **Docker** runs both in a single container (FastAPI starts first, MCP waits for health check)
4. **Benchmark Runner** connects via `docker exec -i` using MCP stdio protocol

## Testing

Unified test runner for retrievers (`apis/retrievers/test_queries.py`):

```
python test_queries.py --mode fastapi address    # Direct HTTP to FastAPI
python test_queries.py --mode mcp address        # In-process MCP -> FastAPI
python test_queries.py --mode docker address     # MCP over stdio in container
```
