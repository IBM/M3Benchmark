# Setup Guide

> **Note:** These steps create a clean environment so your existing branch and code are not affected.

---

## Quick Start — First Time Test

Everything you need to go from a fresh clone to a working single-sample benchmark run:

```bash
# 1. Python environment (Use python or python3)
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[init]"
pip install -r requirements_benchmark.txt

# 2. Set required tokens
export HF_TOKEN=hf_...
export OPENAI_API_KEY=sk-...

# 3. Download data (~30 GB), pull Docker image, start containers
make download
make pull
make start

# 4. Verify all 4 containers are up
docker ps

# 5. Run a single-sample smoke test (Task 1, authors domain)
python benchmark_runner.py --m3_task_id 1 --domain authors --max-samples-per-domain 1 --provider openai

# Results land in output/task_1_<timestamp>/authors.json
```

> If you hit issues, see [Debugging containers](#debugging-containers) below.

---

## Prerequisites

- Docker or Podman running (`docker ps` / `podman ps`)
- If using Podman, alias it: `alias docker=podman`

### Container memory requirements

Task 5 (ChromaDB) requires at least **8 GB** allocated to your container runtime. The default (often 2 GB) will OOM-kill the retriever on startup.

**Docker Desktop**

1. Open **Docker Desktop** → **Settings** → **Resources**
2. Set **Memory** to at least **8 GB**
3. Click **Apply & Restart**

**Podman**

```bash
podman machine stop
podman machine set --memory 8192
podman machine start
```

Verify with: `podman info | grep -i memTotal`

**Rancher Desktop**

1. Open **Rancher Desktop** → **Preferences** → **Virtual Machine**
2. Set **Memory** to at least **8 GB** (8192 MiB)
3. Click **Apply** (the VM will restart)

---

## 1. Clone the Repository

```bash
git clone git@github.ibm.com:AI4BA/enterprise-benchmark.git
cd enterprise-benchmark
```

---

## 2. Create a Python Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[init]"
```

Install benchmark dependencies:

```bash
pip install -r requirements_benchmark.txt
# or manually:
pip install langchain-openai langchain mcp langchain-anthropic langgraph langchain-ollama
```

---

## 3. Download Data, Pull Image, and Start Containers

```bash
# Download benchmark data from HuggingFace (~30 GB)
# You will be prompted for a Hugging Face token
make download

# Pull the m3_environ Docker image
make pull

# Start all four benchmark containers
make start
```

> **Warning:** `make download` fetches ~30 GB of data. This will be reduced in a future release.
> For day-to-day restarts after first-time setup, `docker compose up -d` is a lighter alternative — see [Day-to-day container management](#day-to-day-container-management).

---

## 4. Verify Docker Containers

Once setup completes, confirm four containers are running:

```bash
docker ps
```

You should see **4 containers** listed:

| Container | Purpose |
|-----------|---------|
| `task_1_m3_environ` | Sel/Slot MCP server |
| `task_2_m3_environ` | M3 REST MCP server |
| `task_3_m3_environ` | BPO MCP server + M3 REST API |
| `task_5_m3_environ` | M3 REST API + ChromaDB Retriever |

### Day-to-day container management

After first-time setup, `docker compose` is the simplest way to stop and restart containers:

```bash
# Start all containers (uses locally tagged m3_environ image — no pull)
docker compose up -d

# Start a single container
docker compose up -d task_5_m3_environ

# Stop and remove all containers
docker compose down

# Tail logs for all containers
docker compose logs -f
```

> `docker compose` does **not** pull the latest image automatically. Use `make start` when you want to ensure you're on the current published image (it always pulls first).

### Debugging containers

**Tail logs for all containers at once:**

```bash
make logs
```

**Tail logs for a specific container:**

```bash
docker logs -f task_5_m3_environ
```

**Check memory and CPU usage in real time:**

```bash
docker stats task_1_m3_environ task_2_m3_environ task_3_m3_environ task_5_m3_environ
```

**Inspect a container's environment and config:**

```bash
docker inspect task_5_m3_environ
```

**Open a shell inside a running container:**

```bash
docker exec -it task_5_m3_environ bash
```

**Check if the FastAPI server is responding (Tasks 2, 3, 5):**

```bash
docker exec task_2_m3_environ curl -sf http://localhost:8000/openapi.json | head -c 200
docker exec task_5_m3_environ curl -sf http://localhost:8001/health
```

**Send a test MCP handshake to verify the MCP server is alive:**

```bash
MCP_INIT='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0.1.0"}}}'

echo "$MCP_INIT" | docker exec -i -e MCP_DOMAIN=address task_2_m3_environ python /app/m3-rest/mcp_server.py
echo "$MCP_INIT" | docker exec -i task_3_m3_environ python /app/apis/bpo/mcp/server.py
echo "$MCP_INIT" | docker exec -i -e MCP_DOMAIN=address task_5_m3_environ python /app/retrievers/mcp_server.py
echo "$MCP_INIT" | docker exec -i -e MCP_DOMAIN=superhero task_1_m3_environ python -m apis.m3.python_tools.mcp
```

**Stop and restart a single container:**

```bash
# Simple — uses docker compose
docker compose up -d task_5_m3_environ

# Manual — if you need to override flags
docker rm -f task_5_m3_environ

docker run -d --name task_5_m3_environ \
    --memory=4g \
    -v "$(pwd)/data/db:/app/db:ro" \
    -v "$(pwd)/apis/configs:/app/apis/configs:ro" \
    -v "$(pwd)/data/chroma_data:/app/retrievers/chroma_data" \
    -v "$(pwd)/data/queries:/app/retrievers/queries:ro" \
    m3_environ
```

---

## 5. Run a Benchmark


Set your API keys: (refer to template_env, the benchmark runs across various model providers)

```bash
export OPENAI_API_KEY=<your-key>
export LITELLM_API_KEY=<your-key>
```

Make sure containers are running first:

```bash
make start
```

**By task:**

```bash
# Task 1 — Sel/Slot tools
python benchmark_runner.py --m3_task_id 1 --domain authors --max-samples-per-domain 1 --provider openai

# Task 2 — M3 REST SQL tools
python benchmark_runner.py --m3_task_id 2 --provider openai --domain address
python benchmark_runner.py --m3_task_id 2 --provider openai --domain airline
python benchmark_runner.py --m3_task_id 2     --provider openai          # all domains

# Task 3 — BPO + M3 REST tools
python benchmark_runner.py --m3_task_id 3 --domain airline --provider openai

# Task 5 — ChromaDB retriever
python benchmark_runner.py --m3_task_id 5 --domain address --provider openai
```

**Common options:**

```bash
# Limit samples (good for quick tests)
python benchmark_runner.py --m3_task_id 2 --domain hockey --max-samples-per-domain 5

# Choose provider and model
python benchmark_runner.py --m3_task_id 2 --domain hockey --provider anthropic --model claude-sonnet-4-6
python benchmark_runner.py --m3_task_id 2 --domain hockey --provider openai --model gpt-4o
python benchmark_runner.py --m3_task_id 2 --domain hockey --provider ollama --model llama3.1:8b

# Run multiple tasks in parallel
python benchmark_runner.py --m3_task_id 2 5 --domain address --parallel

# Just list available tools (no agent run)
python benchmark_runner.py --m3_task_id 2 --domain hockey --list-tools

# Limit tools via embedding similarity (top-k)
python benchmark_runner.py --m3_task_id 2 --domain hockey --top-k-tools 10
```

Results are saved to `output/task_{id}_{timestamp}/{domain}.json`.

---

## Integration & Unit Tests

### End-to-end tests (requires HuggingFace token + OpenAI key + Docker)

These tests download data, start containers, run the benchmark pipeline, and validate output.

**Step 1 — Set required env vars:**

```bash
export HF_TOKEN=hf_...
export OPENAI_API_KEY=sk-...
```

Alternatively, use a `.env` file:

```bash
cp template_env .env
# edit .env: set HF_TOKEN and OPENAI_API_KEY
export $(grep -v '^#' .env | xargs)
```

**Step 2 — Run the full e2e suite:**

```bash
make e2e
```

> Containers are started once and shared across all tests. Each task writes to its own isolated output directory.

**Run a single task test (optional):**

```bash
python -m pytest tests/e2e/test_benchmark_e2e.py::TestBenchmarkE2E::test_task1_address -v -s
python -m pytest tests/e2e/test_benchmark_e2e.py::TestBenchmarkE2E::test_task2_address -v -s
python -m pytest tests/e2e/test_benchmark_e2e.py::TestBenchmarkE2E::test_task3_airline -v -s
python -m pytest tests/e2e/test_benchmark_e2e.py::TestBenchmarkE2E::test_task5_address -v -s
```

**Start containers individually (optional — if you need to debug a specific container):**

```bash
# Task 1 — Sel/Slot MCP server
docker run -d --name task_1_m3_environ \
    -v "$(pwd)/data/db:/app/db:ro" \
    -v "$(pwd)/apis/configs:/app/apis/configs:ro" \
    m3_environ

# Task 2 — M3 REST MCP server
docker run -d --name task_2_m3_environ \
    -v "$(pwd)/data/db:/app/db:ro" \
    -v "$(pwd)/apis/configs:/app/apis/configs:ro" \
    m3_environ

# Task 3 — BPO MCP server + M3 REST API
docker run -d --name task_3_m3_environ \
    -v "$(pwd)/data/db:/app/db:ro" \
    -v "$(pwd)/apis/configs:/app/apis/configs:ro" \
    m3_environ

# Task 5 — ChromaDB Retriever (needs extra memory + retriever volumes)
docker run -d --name task_5_m3_environ \
    --memory=4g \
    -v "$(pwd)/data/db:/app/db:ro" \
    -v "$(pwd)/apis/configs:/app/apis/configs:ro" \
    -v "$(pwd)/data/chroma_data:/app/retrievers/chroma_data" \
    -v "$(pwd)/data/queries:/app/retrievers/queries:ro" \
    m3_environ
```

To stop and remove a single container:

```bash
docker rm -f task_2_m3_environ
```

---

### Live MCP connection validation (requires running containers)

```bash
make start    # if containers aren't already running
make validate # connects to all 4 containers and lists tools per domain
```

Or validate a specific domain only:

```bash
python benchmark/validate_clients.py --domain airline
```

---

## Testing the Docker Image

The smoke test validates the image in two tiers — sections 1, 3, and 4 run without any data; section 2 (FastAPI health) requires `data/db` to be populated.

**Without data (just built the image):**

```bash
make test
```

Checks file existence, BPO MCP handshake, and M3 REST MCP handshake. FastAPI health check is skipped with a warning until data is downloaded.

**With data (full validation):**

```bash
make download   # one-time — downloads ~35 GB into data/
make test       # all 4 sections run including FastAPI health
```

**Full first-time setup:**

```bash
make setup      # download → build → test → start → validate
```

| Check | Requires data? | What it verifies |
|-------|---------------|-----------------|
| 1. File existence | No | All required `.py`, `.parquet`, and entrypoint files are in the image |
| 2. M3 REST FastAPI | Yes (`data/db`) | `/openapi.json` responds and has at least one route |
| 3. BPO MCP handshake | No | `python /app/apis/bpo/mcp/server.py` returns a valid JSON-RPC response |
| 4. M3 REST MCP handshake | Yes (`data/db`) | `python /app/m3-rest/mcp_server.py` returns a valid JSON-RPC response (requires FastAPI on port 8000) |

---

## Makefile Targets

| Target | What it does |
|--------|-------------|
| `make download` | `python m3_setup.py --download-data` — syncs all 4 HuggingFace repos into `data/` |
| `make pull` | Pull the `m3_environ` image from Docker Hub |
| `make build` | Build the Docker image |
| `make test` | Smoke-test the image (file checks + MCP handshakes) |
| `make validate` | Live MCP connection check against running containers |
| `make tag` | Tag image for Docker Hub |
| `make push` | Push image to Docker Hub |
| `make release` | `build → test → tag → push` |
| `make setup` | `download → build → test → start → validate` — full first-time setup |
| `make start` | Start all benchmark containers |
| `make stop` | Stop and remove all benchmark containers |
| `make clean` | Stop containers and remove the local `m3_environ` Docker image |
| `make e2e` | Run end-to-end benchmark tests (requires `HF_TOKEN` + `OPENAI_API_KEY`) |
| `make e2e-quick` | Run e2e tests against already-running containers — skips download and container restart (requires `OPENAI_API_KEY` only) |
| `make logs` | Last 20 log lines per container |

---

## Rebuilding and Pushing the Docker Image (to be used by M3 ENV maintainers only)

Run these commands from the **project root** whenever `docker/Dockerfile.unified`, `apis/bpo/`, or any other server code changes.

### 0. Log in to Docker Hub (one-time)

```bash
docker login docker.io --username
# Username: amurthi44g1wd
# Password: <your Docker Hub password or access token>
```

> **Using an access token?** The token must have **Read & Write** (or **Read, Write & Delete**) scope — a read-only token will cause `insufficient scopes` errors on push.
>
> Create one at **Docker Hub → Account Settings → Personal access tokens → Generate new token**, set Access permissions to **Read & Write**, then use the token as the password above.

### 1. One-command release (recommended)

```bash
make release
```

This runs `build → test → tag → push` in sequence and stops on the first failure.

### 2. Step by step

Ignore the below four steps, if you already did a `make release`

```bash
make build      # docker build -t m3_environ -f docker/Dockerfile.unified .
make test       # smoke-test: file checks + M3 REST health + BPO/M3 MCP handshakes
make tag        # docker tag m3_environ docker.io/amurthi44g1wd/m3_environ:latest
make push       # docker push docker.io/amurthi44g1wd/m3_environ:latest
```

After restarting containers you can also run a live connection check against all four containers:

```bash
make start      # restart containers with new image
make validate   # python benchmark/validate_clients.py — tests every MCP server
```

### 3. Restart containers to pick up the new image

```bash
make start
```

This stops/removes all existing benchmark containers, pulls the freshly pushed image, and restarts all four containers.
