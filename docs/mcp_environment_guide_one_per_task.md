# MCP Environment Guide: One Container Per Task

This guide covers the **two-container** approach where each task runs in its own isolated container, both from the same `m3_environ` Docker image. This provides memory isolation and crash protection between tasks.

> For the single-container approach (simpler, less isolation), see [mcp_environment_guide_single_unified_image.md](mcp_environment_guide_single_unified_image.md).

## Why Two Containers?

| Concern | Single Container | One Per Task |
|---------|-----------------|--------------|
| Memory isolation | Shared | Each task gets its own memory budget |
| Crash protection | One crash kills all tasks | Task 2 crash doesn't affect Task 5 |
| Parallel execution | Shared CPU/memory | Independent resource limits possible |
| Startup time | One container, all services | Task 2 container is lean (REST only); Task 5 still starts both services |
| Simplicity | Simpler | Slightly more setup |

## Docker Image

Both containers use the **same image** — build it once from source:

```bash
make build
```

## Container Configurations

### `capability_2_dashboard_apis_m3_environ` — M3 SQL Tools

Runs **only M3 REST FastAPI** (port 8000). The retriever is not started because `chroma_data` is not mounted.

```bash
docker run -d --name capability_2_dashboard_apis_m3_environ \
    -v "$(pwd)/apis/m3/rest/db:/app/db:ro" \
    -v "$(pwd)/apis/configs:/app/apis/configs:ro" \
    m3_environ
```

**Services started:** M3 REST FastAPI (:8000)
**Memory footprint:** ~200 MB

### `capability_4_multiturn_m3_environ` — Retriever Tools + M3 REST

Runs **both M3 REST FastAPI** (port 8000) **and Retriever FastAPI** (port 8001). Task 5 needs M3 REST because some retriever workflows depend on it.

```bash
docker run -d --name capability_4_multiturn_m3_environ \
    -v "$(pwd)/apis/m3/rest/db:/app/db:ro" \
    -v "$(pwd)/apis/configs:/app/apis/configs:ro" \
    -v "$(pwd)/apis/retrievers/chroma_data:/app/retrievers/chroma_data" \
    -v "$(pwd)/apis/retrievers/queries:/app/retrievers/queries:ro" \
    m3_environ
```

**Services started:** M3 REST FastAPI (:8000) + Retriever FastAPI (:8001)
**Memory footprint:** ~2-3 GB (embeddings model loaded)

> **How it works:** The entrypoint script automatically detects whether `chroma_data/` is mounted. If present, the retriever starts. If absent, it's skipped. No extra configuration needed — the volume mounts control the behavior.

## No Port Collisions

Both containers run M3 REST on internal port 8000, but there is **no collision**. Each container has its own network namespace — the internal ports are isolated. MCP servers inside the container talk to `localhost:8000` within their own container.

No `-p` port mapping flags are needed unless you want to debug the FastAPI servers from the host:

```bash
# Optional: expose ports for debugging (use different host ports to avoid collision)
docker run -d --name capability_2_dashboard_apis_m3_environ -p 8000:8000 ...
docker run -d --name capability_4_multiturn_m3_environ -p 9000:8000 -p 9001:8001 ...
```

## Task Reference

| Task ID | Container | MCP Exec Command | Running Services |
|---------|-----------|-----------------|------------------|
| 2 | `capability_2_dashboard_apis_m3_environ` | `docker exec -i -e MCP_DOMAIN=<domain> capability_2_dashboard_apis_m3_environ python /app/m3-rest/mcp_server.py` | M3 REST (:8000) → SQLite |
| 5 | `capability_4_multiturn_m3_environ` | `docker exec -i -e MCP_DOMAIN=<domain> capability_4_multiturn_m3_environ python /app/retrievers/mcp_server.py` | M3 REST (:8000) → SQLite **+** Retriever (:8001) → ChromaDB |

## Environment Variable: `MCP_DOMAIN`

The only environment variable you need is `MCP_DOMAIN`, passed via `docker exec -e`:

```bash
# Task 2 — query the "hockey" domain
docker exec -i -e MCP_DOMAIN=hockey capability_2_dashboard_apis_m3_environ python /app/m3-rest/mcp_server.py

# Task 5 — query the "address" domain
docker exec -i -e MCP_DOMAIN=address capability_4_multiturn_m3_environ python /app/retrievers/mcp_server.py
```

## Volume Mounts

### capability_2_dashboard_apis_m3_environ

| Host Path | Container Path | Mode | Contents |
|-----------|---------------|------|----------|
| `apis/m3/rest/db/` | `/app/db/` | `ro` | SQLite databases |
| `apis/configs/` | `/app/apis/configs/` | `ro` | MCP configuration files |

### capability_4_multiturn_m3_environ

| Host Path | Container Path | Mode | Contents |
|-----------|---------------|------|----------|
| `apis/m3/rest/db/` | `/app/db/` | `ro` | SQLite databases |
| `apis/configs/` | `/app/apis/configs/` | `ro` | MCP configuration files |
| `apis/retrievers/chroma_data/` | `/app/retrievers/chroma_data/` | **rw** | ChromaDB vector collections |
| `apis/retrievers/queries/` | `/app/retrievers/queries/` | `ro` | Retriever query configurations |

> **Note:** `chroma_data` must be mounted read-write. ChromaDB uses SQLite WAL mode internally and needs write access even for read operations.

## How It Works

```
Your Agent (benchmark_runner_one_per_task.py)
  │
  ├── Task 2 queries ──► capability_2_dashboard_apis_m3_environ
  │     │                  Services: M3 REST (:8000) only
  │     └── docker exec ... python /app/m3-rest/mcp_server.py
  │           └── localhost:8000 (M3 REST) → SQLite databases
  │
  └── Task 5 queries ──► capability_4_multiturn_m3_environ
        │                  Services: M3 REST (:8000) + Retriever (:8001)
        └── docker exec ... python /app/retrievers/mcp_server.py
              ├── localhost:8001 (Retriever) → ChromaDB collections
              └── localhost:8000 (M3 REST) → SQLite databases
```

## Connecting Your Agent (MCP Python SDK)

```python
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client

# Task 2 — points to capability_2_dashboard_apis_m3_environ container
task2_params = StdioServerParameters(
    command="docker",
    args=[
        "exec", "-i",
        "-e", f"MCP_DOMAIN={domain}",
        "capability_2_dashboard_apis_m3_environ",
        "python", "/app/m3-rest/mcp_server.py",
    ],
)

# Task 5 — points to capability_4_multiturn_m3_environ container
task5_params = StdioServerParameters(
    command="docker",
    args=[
        "exec", "-i",
        "-e", f"MCP_DOMAIN={domain}",
        "capability_4_multiturn_m3_environ",
        "python", "/app/retrievers/mcp_server.py",
    ],
)
```

## Benchmark Runner: `benchmark_runner_one_per_task.py`

The runner automatically manages containers and routes each task to the right one:

```bash
# Task 2 only
python benchmark_runner_one_per_task.py --capability_id 2 --run-agent --domain address

# Task 5 only
python benchmark_runner_one_per_task.py --capability_id 5 --run-agent --domain address

# Both tasks in parallel (each in its own container)
python benchmark_runner_one_per_task.py --capability_id 2 5 --run-agent --domain address --parallel

# List tools
python benchmark_runner_one_per_task.py --capability_id 5 --list-tools --domain address
```

### What the runner does

1. **Start containers** — for each requested capability_id, starts the corresponding container (if not already running) with the right volume mounts
2. **Wait for readiness** — polls container logs until "All services running" appears
3. **Run benchmark** — executes `docker exec` MCP sessions against the correct container
4. **Stop containers** — optionally stops containers when done (or leaves them running for reuse)

### Container lifecycle

| Flag | Behavior |
|------|----------|
| Default | Assumes containers are already running |
| `--start-containers` | Auto-starts required containers with correct volume mounts before running |
| `--cleanup` | Stops and removes containers after benchmark completes |

### Task-to-container mapping

| Task ID | Container Name | Services | Volume Mounts |
|---------|---------------|----------|---------------|
| 2 | `capability_2_dashboard_apis_m3_environ` | M3 REST only | `db`, `configs` |
| 5 | `capability_4_multiturn_m3_environ` | M3 REST + Retriever | `db`, `configs`, `chroma_data`, `queries` |

## Quick Start (Full Walkthrough)

```bash
# 1. Build the image
make build

# 2. Start all containers
docker compose up -d

# 3. Wait for readiness (~30s for capability_2, ~60-120s for capability_4)
docker logs -f capability_2_dashboard_apis   # wait for "All services running"
docker logs -f capability_4_multiturn        # wait for "All services running"

# 4. Verify
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0.1.0"}}}' \
  | docker exec -i -e MCP_DOMAIN=address capability_2_dashboard_apis python /app/m3-rest/mcp_server.py

echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0.1.0"}}}' \
  | docker exec -i -e MCP_DOMAIN=address capability_4_multiturn python /app/retrievers/mcp_server.py

# 5. Run benchmark
python benchmark_runner.py --capability_id 2 4 --run-agent --domain address --parallel \
    --provider openai --model gpt-4o
```

## Cleanup

```bash
docker compose down
```

## Resource Limits (Optional)

Since each task runs in its own container, you can set per-task resource limits:

```bash
# Task 2 — lightweight, limit to 512 MB
docker run -d --name capability_2_dashboard_apis_m3_environ --memory=512m \
    -v "$(pwd)/apis/m3/rest/db:/app/db:ro" \
    -v "$(pwd)/apis/configs:/app/apis/configs:ro" \
    m3_environ

# Task 5 — heavy (embeddings model), give it 4 GB
docker run -d --name capability_4_multiturn_m3_environ --memory=4g \
    -v "$(pwd)/apis/m3/rest/db:/app/db:ro" \
    -v "$(pwd)/apis/configs:/app/apis/configs:ro" \
    -v "$(pwd)/apis/retrievers/chroma_data:/app/retrievers/chroma_data" \
    -v "$(pwd)/apis/retrievers/queries:/app/retrievers/queries:ro" \
    m3_environ
```
