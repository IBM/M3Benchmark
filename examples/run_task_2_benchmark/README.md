# Task 2 Benchmark Runner

Simple runner that iterates over all 83 domains, starts a per-domain MCP server
via `docker exec` (stdio), lists the available tools, calls the agent, then stops
the server before moving to the next domain.

## Files

| File | Description |
|------|-------------|
| `run_task2.py` | Main runner script |
| `task2_server.yaml` | MCP server config (container, env vars, command, domain list) |

---

## Prerequisites

### 1. Install dependencies

From the repo root:

```bash
pip install -e ".[init]"
```

### 2. Download benchmark data

```bash
python m3_setup.py --download-data
```

This downloads databases, task configs, and retriever data from HuggingFace into `data/`.
You will be prompted for a HuggingFace token if `HF_TOKEN` is not set in your environment.

### 3. Pull the Docker image and start containers

```bash
python m3_setup.py
```

This will:
- Stop and remove any existing benchmark containers
- Pull the latest `m3_environ` image from Docker Hub
- Start `task_2_m3_environ` (and the other task containers)
- Wait until all containers are healthy

To start containers only (skipping data download):

```bash
python m3_setup.py --start-containers
```

To stop containers when you're done:

```bash
python m3_setup.py --stop-containers
```

---

## Running the Task 2 runner

From this directory (`examples/run_task_2_benchmark/`):

```bash
# Run all 83 domains
python run_task2.py

# Run a single domain (useful for quick testing)
python run_task2.py --domain airline

# Use podman instead of docker
python run_task2.py --runtime podman
```

### What it does per domain

1. Builds `docker exec -i -e MCP_DB_ROOT=/app/db -e MCP_DOMAIN=<domain> task_2_m3_environ python /app/m3-rest/mcp_server.py`
2. Opens an MCP session over stdio
3. Lists all tools with their descriptions and parameters
4. Calls the agent *(placeholder — wire in your agent here)*
5. Waits for the agent response *(placeholder)*
6. Closes the session — the server subprocess is stopped automatically

### Wiring in an agent

Find the two placeholder comments in `run_task2.py`:

```python
# --- Agent call placeholder ---
# result = await agent.run(domain=domain, tools=tools, session=session)
print("\n[placeholder] Agent called with tools")

# --- Wait for agent response placeholder ---
# answer = await result
print("[placeholder] Agent response received")
```

Replace them with your agent call. `tools` is the list of `mcp.types.Tool` objects
and `session` is the live `ClientSession` you can pass directly to the agent.

---

## Configuration

`task2_server.yaml` controls everything about how the MCP server is started:

```yaml
task_2:
  container: task_2_m3_environ   # Docker container name
  env:
    MCP_DB_ROOT: /app/db         # Path to databases inside the container
  command:
    - python
    - /app/m3-rest/mcp_server.py # MCP server entry point inside the container
  domains:
    - address
    - airline
    - ...                        # Full list of 83 domains
```

To run only a subset of domains, either use `--domain <name>` on the CLI or edit
the `domains` list in the YAML.
