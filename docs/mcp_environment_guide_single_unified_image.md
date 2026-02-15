# MCP Environment Guide

This guide covers how to set up and run the MCP tool servers that power the M3 Benchmark. All tasks use a single Docker image (`m3_environ`) that bundles every server you need.

## Docker Image

| Image | Docker Hub |
|-------|-----------|
| `m3_environ` | `docker.io/amurthi44g1wd/m3_environ:latest` |

## Quick Start

```bash
# 1. Pull the image
docker pull docker.io/amurthi44g1wd/m3_environ:latest
docker tag docker.io/amurthi44g1wd/m3_environ:latest m3_environ

# 2. Start the container
docker run -d --name m3_environ \
    -v "$(pwd)/apis/m3/rest/db:/app/db:ro" \
    -v "$(pwd)/apis/configs:/app/apis/configs:ro" \
    -v "$(pwd)/apis/retrievers/chroma_data:/app/retrievers/chroma_data" \
    -v "$(pwd)/apis/retrievers/queries:/app/retrievers/queries:ro" \
    m3_environ

# 3. Verify (wait ~30s for startup)
docker logs m3_environ 2>&1 | tail -5
# Should show: "=== All services running. Container ready for exec. ==="
```

## How It Works

The container starts two internal FastAPI servers at boot. Your agent never talks to these directly — instead, it communicates with thin MCP servers spawned on demand via `docker exec`. Each MCP server translates [MCP protocol](https://modelcontextprotocol.io/) tool calls into HTTP requests to its corresponding FastAPI backend.

```
Your Agent
  │
  └── docker exec -i -e MCP_DOMAIN=<domain> m3_environ <command>
        │
        ├── Task 2 command ──► localhost:8000 (M3 REST) ──► SQLite databases
        └── Task 5 command ──► localhost:8001 (Retriever) ──► ChromaDB collections
```

## Task Reference

| Task ID | Name | Docker Exec Command | Backend |
|---------|------|---------------------|---------|
| 2 | M3 SQL Tools | `docker exec -i -e MCP_DOMAIN=<domain> m3_environ python /app/m3-rest/mcp_server.py` | FastAPI :8000 → SQLite |
| 5 | Retriever Tools | `docker exec -i -e MCP_DOMAIN=<domain> m3_environ python /app/retrievers/mcp_server.py` | FastAPI :8001 → ChromaDB |

## Environment Variable: `MCP_DOMAIN`

The only environment variable you need to set is `MCP_DOMAIN`. It is passed via the `-e` flag on `docker exec` and tells the MCP server which domain's data to load.

| Variable | Where | How to set | Example values |
|----------|-------|-----------|----------------|
| `MCP_DOMAIN` | Passed into container via `docker exec -e` | `-e MCP_DOMAIN=address` | `address`, `hockey`, `airline`, `superhero`, `finance`, `music` |

Each domain corresponds to a dataset. The MCP server loads the relevant SQLite database (task 2) or ChromaDB collection (task 5) based on this value.

**Full example:**

```bash
# Task 2 — load the "hockey" domain and start an MCP session
docker exec -i -e MCP_DOMAIN=hockey m3_environ python /app/m3-rest/mcp_server.py

# Task 5 — load the "address" domain and start an MCP session
docker exec -i -e MCP_DOMAIN=address m3_environ python /app/retrievers/mcp_server.py
```

## Connecting Your Agent

The MCP servers use **stdio** transport. Your agent should:

1. Spawn `docker exec -i -e MCP_DOMAIN=<domain> m3_environ <command>` as a subprocess
2. Send JSON-RPC messages to its stdin
3. Read JSON-RPC responses from its stdout

If you're using the MCP Python SDK:

```python
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(
    command="docker",
    args=[
        "exec", "-i",
        "-e", f"MCP_DOMAIN={domain}",
        "m3_environ",
        "python", "/app/m3-rest/mcp_server.py",  # or /app/retrievers/mcp_server.py
    ],
)

async with stdio_client(server_params) as (read, write):
    # Use read/write streams with MCP ClientSession
    ...
```

## Volume Mounts

| Host Path | Container Path | Mode | Contents |
|-----------|---------------|------|----------|
| `apis/m3/rest/db/` | `/app/db/` | `ro` | SQLite databases (one per domain) |
| `apis/configs/` | `/app/apis/configs/` | `ro` | MCP configuration files |
| `apis/retrievers/chroma_data/` | `/app/retrievers/chroma_data/` | **rw** | ChromaDB vector collections |
| `apis/retrievers/queries/` | `/app/retrievers/queries/` | `ro` | Retriever query configurations |

> **Note:** `chroma_data` must be mounted read-write. ChromaDB uses SQLite WAL mode internally and needs write access even for read operations.

## Verifying the Setup

**Check both FastAPI backends are healthy:**

```bash
# Add -p 8000:8000 -p 8001:8001 to your docker run if you want to curl from the host
curl http://localhost:8000/openapi.json | head -c 200
curl http://localhost:8001/health
```

**Test MCP servers with a raw JSON-RPC `initialize` message:**

```bash
# Task 2
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0.1.0"}}}' \
  | docker exec -i -e MCP_DOMAIN=address m3_environ python /app/m3-rest/mcp_server.py

# Task 5
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0.1.0"}}}' \
  | docker exec -i -e MCP_DOMAIN=address m3_environ python /app/retrievers/mcp_server.py
```

A successful response looks like: `{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05",...}}`

## Using the Benchmark Runner

The provided `benchmark_runner_single_docker_image.py` handles all of the above automatically:

```bash
# Task 2
python benchmark_runner_single_docker_image.py \
    --task_id 2 --run-agent --domain address \
    --provider openai --model gpt-4o

# Task 5
python benchmark_runner_single_docker_image.py \
    --task_id 5 --run-agent --domain address \
    --provider openai --model gpt-4o

# Both tasks in parallel
python benchmark_runner_single_docker_image.py \
    --task_id 2 5 --run-agent --domain address --parallel \
    --provider openai --model gpt-4o

# List available MCP tools (no LLM needed)
python benchmark_runner_single_docker_image.py \
    --task_id 2 --list-tools --domain address
```

Results are saved to `output/task_{id}_{timestamp}/<domain>.json`.

## Building from Source (optional)

If you need to modify the servers or rebuild the image:

```bash
docker build -t m3_environ -f docker/Dockerfile.unified .
```

## Cleanup

```bash
docker stop m3_environ && docker rm m3_environ
```
