# Debugging Guide

Tips for inspecting containers, reading logs, and filtering MCP server output.

---

## Container health

**lazydocker — interactive terminal UI (recommended)**

```bash
brew install lazydocker   # install once
lazydocker                # run from the project directory
```

Gives you a live view of all containers, logs, and resource stats in one terminal window. Run it from the project root so it picks up `docker-compose.yml` automatically.

**Tail logs for all containers at once:**

```bash
make logs
```

**Tail logs for a specific container:**

```bash
docker logs -f capability_4_multiturn_m3_environ
```

**Check memory and CPU usage in real time:**

```bash
docker stats capability_1_bi_apis_m3_environ capability_2_dashboard_apis_m3_environ capability_3_multihop_reasoning_m3_environ capability_4_multiturn_m3_environ
```

**Inspect a container's environment and config:**

```bash
docker inspect capability_4_multiturn_m3_environ
```

**Open a shell inside a running container:**

```bash
docker exec -it capability_4_multiturn_m3_environ bash
```

**Check if the FastAPI server is responding (Tasks 2, 3, 5):**

```bash
docker exec capability_2_dashboard_apis_m3_environ curl -sf http://localhost:8000/openapi.json | head -c 200
docker exec capability_4_multiturn_m3_environ curl -sf http://localhost:8001/health
```

**Send a test MCP handshake to verify the MCP server is alive:**

```bash
MCP_INIT='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0.1.0"}}}'

echo "$MCP_INIT" | docker exec -i -e MCP_DOMAIN=address capability_2_dashboard_apis_m3_environ python /app/m3-rest/mcp_server.py
echo "$MCP_INIT" | docker exec -i capability_3_multihop_reasoning_m3_environ python /app/apis/bpo/mcp/server.py
echo "$MCP_INIT" | docker exec -i -e MCP_DOMAIN=address capability_4_multiturn_m3_environ python /app/retrievers/mcp_server.py
echo "$MCP_INIT" | docker exec -i -e MCP_DOMAIN=superhero capability_1_bi_apis_m3_environ python -m apis.m3.python_tools.mcp
```

**Stop and restart a single container:**

```bash
# Simple — uses docker compose
docker compose up -d capability_4_multiturn_m3_environ

# Manual — if you need to override flags
docker rm -f capability_4_multiturn_m3_environ

docker run -d --name capability_4_multiturn_m3_environ \
    --memory=4g \
    -v "$(pwd)/data/databases:/app/db:ro" \
    -v "$(pwd)/apis/configs:/app/apis/configs:ro" \
    -v "$(pwd)/indexed_documents:/app/retrievers/chroma_data" \
    -v "$(pwd)/data/queries:/app/retrievers/queries:ro" \
    m3_environ
```

---

## Benchmark run logs

Each benchmark run writes a timestamped `run.log` alongside its results:

```
output/
  task_2_feb_24_10_30am/
    hockey.json        ← scored results
    address.json
    run.log            ← full human-readable run transcript with timestamps
```

```bash
# Tail a run as it happens
tail -f output/task_2_*/run.log

# Read a completed run
cat output/task_2_feb_24_10_30am/run.log

# Quick summary — grep for status lines only
grep "Status:" output/task_2_feb_24_10_30am/run.log

# Find all errors across a run
grep "error\|Error\|timeout" output/task_2_feb_24_10_30am/run.log
```

---

## MCP server logs with jq

The MCP servers (Tasks 1, 2, 3, 5) emit JSON lines to **stderr** during each
`docker exec` call. These flow to the terminal where `benchmark_runner.py` is
running — they do **not** appear in `docker logs <container>`.

Redirect stderr to a file to capture and filter them:

```bash
# Capture MCP server logs separately from benchmark runner stdout
python benchmark_runner.py --capability_id 2 --domain hockey \
    --provider openai 2>mcp.log

# Or capture everything together (stdout + stderr)
python benchmark_runner.py --capability_id 2 --domain hockey \
    --provider openai 2>&1 | tee full.log
```

**jq recipes** (`brew install jq` if not installed):

> `mcp.log` contains a mix of JSON lines (MCP server) and plain-text lines
> (benchmark runner's own logger). Always pre-filter with `grep '^{'` to pass
> only valid JSON to jq.

```bash
# See every tool the agent called, in order
grep '^{' mcp.log | jq -r 'select(.msg | startswith("tool_call")) | .ts + "  " + .msg'

# Filter to a specific domain
grep '^{' mcp.log | jq 'select(.domain == "hockey")'

# Errors only
grep '^{' mcp.log | jq 'select(.level == "ERROR")'

# Startup and init messages (what the server found at startup)
grep '^{' mcp.log | jq 'select(.msg | test("Discovered|tools|Starting"))'

# One-liner summary: timestamp + level + message
grep '^{' mcp.log | jq -r '[.ts, .level, .domain, .msg] | @tsv'

# Count tool calls per tool name
grep '^{' mcp.log | jq -r 'select(.msg | startswith("tool_call")) | .msg' \
    | sort | uniq -c | sort -rn
```

**In parallel mode** (`--parallel`), logs from all tasks are interleaved. Filter
by `capability_id` to isolate a single task:

```bash
grep '^{' mcp.log | jq 'select(.capability_id == "2")'
grep '^{' mcp.log | jq 'select(.capability_id == "5" and .level == "ERROR")'
```

**Container FastAPI logs** (long-lived service, separate from MCP server logs):

> **Note:** `docker logs <container>` shows the FastAPI/uvicorn service logs
> (human-readable text, **not JSON**). Do **not** pipe these through `jq` — it
> will fail with a parse error. Use `jq` only on `mcp.log` captured from the
> benchmark runner's stderr (see above).

```bash
# These show FastAPI request logs, SQLite errors, startup issues
docker logs capability_2_dashboard_apis_m3_environ 2>&1 | tail -50
docker logs -f capability_4_multiturn_m3_environ   # follow in real time
```
