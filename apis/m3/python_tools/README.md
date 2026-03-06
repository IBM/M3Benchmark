# M3 Python Tools MCP Server

An MCP (Model Context Protocol) server that exposes slot-filling and selection data manipulation tools for AI agents. Supports running locally as a subprocess or inside a Docker/Podman container.

**Docker image:** `docker.io/amurthi44g1wd/m3-sel-slot-server:latest`

## Quick Test

Run the published Docker image and verify the MCP server works end-to-end.

### 1. Download databases

The server needs the Bird-Bench SQLite databases. Download train and dev sets from https://bird-bench.github.io/, unzip, and place the database folders under a `db/` directory.

### 2. Start the container

```bash
docker run -d --name m3-sel-slot-server \
    -v "$(pwd)/apis/m3/rest/db:/app/db:ro" \
    -v "$(pwd)/apis/configs:/app/apis/configs:ro" \
    docker.io/amurthi44g1wd/m3-sel-slot-server:latest
```

### 3. Verify it's running

```bash
docker ps --filter name=m3-sel-slot-server
```

### 4. Test MCP server manually

```bash
# List tools (the server will start, print tools to stderr, then wait for MCP input)
docker exec -i -e MCP_DOMAIN=superhero m3-sel-slot-server \
    python -m apis.m3.python_tools.mcp 2>&1 | head -20
```

### 5. Run with an LLM agent

```bash
# From the project root
python examples/demo.py --mode docker \
    --container-name m3-sel-slot-server --domain superhero \
    --provider openai --model gpt-4o \
    --max-samples-per-domain 1
```

### 6. Stop

```bash
docker stop m3-sel-slot-server && docker rm m3-sel-slot-server
```

## Quick Start

### Local (subprocess)

```bash
# From the project root
MCP_DOMAIN=superhero python -m apis.m3.python_tools.mcp
```

### Docker (build from source)

```bash
# 1. Build (from apis/m3/python_tools/)
cd apis/m3/python_tools
docker build -t m3-sel-slot-server .

# 2. Run (from the project root — mounts databases + configs)
cd /path/to/enterprise-benchmark-1
docker run -d --name m3-sel-slot-server \
    -v "$(pwd)/apis/m3/rest/db:/app/db:ro" \
    -v "$(pwd)/apis/configs:/app/apis/configs:ro" \
    m3-sel-slot-server

# 3. Exec into the running container (MCP STDIO protocol)
docker exec -i -e MCP_DOMAIN=superhero m3-sel-slot-server \
    python -m apis.m3.python_tools.mcp
```

## Demo

The `examples/demo.py` script connects an LLM agent to the MCP server and runs benchmark queries.

```bash
# Local subprocess (default)
MCP_DOMAIN=superhero python examples/demo.py \
    --provider openai --model gpt-4o \
    --max-samples-per-domain 1

# Docker container
python examples/demo.py --mode docker \
    --container-name m3-sel-slot-server --domain superhero \
    --provider openai --model gpt-4o \
    --max-samples-per-domain 1

# WebSocket (connect to an externally running server)
python examples/demo.py --mode websocket \
    --server-url ws://localhost:8000/mcp \
    --provider openai --model gpt-4o
```

### Demo Options

| Flag | Description |
|------|-------------|
| `--mode` | `stdio` (default), `docker`, or `websocket` |
| `--provider` | `openai`, `anthropic`, `ollama`, `rits`, `litellm`, `watsonx` |
| `--model` | Model name (default: provider-specific) |
| `--max-samples-per-domain` | Limit queries to process |
| `--container-name` | Docker container name (default: `m3-sel-slot-server`) |
| `--container-runtime` | `docker` or `podman` (default: auto-detect) |
| `--domain` | Domain for `MCP_DOMAIN` (default: `superhero`) |
| `--server-url` | WebSocket URL (required for `--mode websocket`) |

## Database Path Resolution

The server constructs the database path as `db/{domain}/{domain}.sqlite` from the `MCP_DOMAIN` environment variable (default: `superhero`). It searches the following locations in order, using the first match:

| Priority | Path | Typical use case |
|----------|------|------------------|
| 1 | `db/{domain}/{domain}.sqlite` | Docker container (`/app/db/` volume mount) |
| 2 | `data/db/{domain}/{domain}.sqlite` | Alternative local layout |
| 3 | `apis/m3/rest/db/{domain}/{domain}.sqlite` | Local runs from project root |

This means the server works out of the box in both Docker and local environments without any path configuration.

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `MCP_DOMAIN` | Database domain name (e.g. `superhero`, `hockey`) |
| `MCP_CACHE_DIR` | Cache directory for IO operations |
| `MCP_TABLES` | Comma-separated list of tables to load |
| `MCP_SERVER_TYPE` | `router` (default), `slot_filling`, or `selection` |
| `MCP_TRANSPORT` | `stdio` (default) or `websocket` |
| `MCP_HOST` | WebSocket host (default: `127.0.0.1`) |
| `MCP_PORT` | WebSocket port (default: `8000`) |

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--db PATH` | Path to SQLite database file |
| `--tables TABLE1,TABLE2` | Comma-separated list of tables to load (default: all) |
| `--config FILE` | Path to JSON config file |
| `--cache-dir DIR` | Directory for caching (default: `./mcp_cache`) |
| `--mode` | Server type: `router` (default), `slot_filling`, or `selection` |
| `--transport` | `stdio` (default) or `websocket` |
| `--host` | WebSocket bind host |
| `--port` | WebSocket port |
| `-v, --verbose` | Enable verbose logging |

### Config File

Create `slot-filling-mcp.json` in the working directory or `~/.config/slot-filling-mcp/config.json`:

```json
{
    "database_path": "/path/to/database.sqlite",
    "tables": ["users", "orders"],
    "cache_dir": "./cache"
}
```

Configuration priority (highest to lowest):
1. Command line arguments
2. Environment variables
3. Config file
4. Default values

## Docker Container Details

### Building and Pushing

```bash
# Build (from apis/m3/python_tools/)
cd apis/m3/python_tools
docker build -t m3-sel-slot-server .

# Tag for registry
docker tag m3-sel-slot-server docker.io/amurthi44g1wd/m3-sel-slot-server:latest

# Push
docker push docker.io/amurthi44g1wd/m3-sel-slot-server:latest
```

The image contains only the Python code and dependencies (~200 MB). Databases (32 GB, 84 domains) and universe configs are mounted at runtime.

### Running

```bash
docker run -d --name m3-sel-slot-server \
    -v "/path/to/db:/app/db:ro" \
    -v "/path/to/apis/configs:/app/apis/configs:ro" \
    docker.io/amurthi44g1wd/m3-sel-slot-server:latest
```

| Volume mount | Container path | Contents |
|-------------|----------------|----------|
| `apis/m3/rest/db/` | `/app/db/` | 84 SQLite databases (one per domain) |
| `apis/configs/` | `/app/apis/configs/` | `mcp_init_mapping.json` (universe configs) |

The container runs `tail -f /dev/null` to stay alive for `docker exec -i` usage.

### Stopping / Removing

```bash
docker stop m3-sel-slot-server && docker rm m3-sel-slot-server
```

## Available Tools

### Data Manipulation Tools

| Tool | Description |
|------|-------------|
| `filter_data` | Filter data based on conditions (equal_to, greater_than, like, etc.) |
| `sort_data` | Sort data by a key (ascending/descending) |
| `aggregate_data` | Perform aggregations (min, max, sum, count, mean, std) |
| `transform_data` | Apply transformations (substring, abs, datetime) |
| `retrieve_data` | Select specific columns from data |
| `concatenate_data` | Combine two datasets |
| `select_unique_values` | Get unique values from an array |
| `peek_fcn` | Inspect data structure and content |
| `get_data` | Get current data or switch to a different tool universe |

### Helper Tools

| Tool | Description |
|------|-------------|
| `list_tables` | List all preloaded tables with their columns and row counts |
| `get_table` | Get the full data from a specific table |

## Column Naming Convention

When tables are loaded, column names are prefixed with the table name to avoid collisions:
- Table `users` with column `id` becomes `users_id`
- Table `orders` with column `id` becomes `orders_id`

## Programmatic Usage

```python
import asyncio
from apis.m3.python_tools.mcp import create_server, MCPServerConfig

config = MCPServerConfig(
    database_path="apis/m3/rest/db/superhero/superhero.sqlite",
    tables=["superhero", "hero_power", "superpower"],
)

server = create_server(config)
asyncio.run(server.run())
```
