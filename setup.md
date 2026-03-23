# Setup Guide

> **Note:** These steps create a clean environment so your existing branch and code are not affected.

---

## Quick Start — First Time Test

Choose one of the two routes below. Both end at the same place: 4 containers running and a passing smoke test.

---

### Option 1 — Build from source + Docker Compose

Use this when you've cloned the repo and want to build and run everything locally.

> **Podman users:** `docker compose` is a Docker plugin and won't exist on Podman-only systems. Either alias it (`alias docker=podman`) or replace `docker compose` with `podman compose` throughout. `make` targets use `$(DOCKER)` and auto-detect the runtime, so they work without aliasing.

`docker compose` can be installed from: https://github.com/docker/compose 
Depending on the OS (RHEL/Debian/Ubuntu), starting the podman socket may be required: `systemctl --user enable --now podman.socket`

```bash
# 1. Python environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements_benchmark.txt

# 2. Download benchmark data (~30 GB)
make download

# 3. Stop any existing containers, build the image, and start all 4 containers
docker compose down    # or: podman compose down
make build
docker compose up -d   # or: podman compose up -d

# 4. Wait ~60 s for internal services to initialize, then verify
docker compose ps      # or: podman compose ps

# 5. Run a single-sample smoke test (Task 1, authors domain)

# 2. (optional) Set required tokens based on the LLM Provided

# Optional step based on the LLM provider you would like to use
export OPENAI_API_KEY=sk-...

python benchmark_runner.py --m3_capability_id 1 --domain california_schools --max-samples-per-domain 1 --provider openai

# Results land in output/task_1_<timestamp>/authors.json
```

After making changes to server code or `docker/Dockerfile.unified`, re-run `make build` then `docker compose up -d` (or `podman compose up -d`) to pick them up.

```bash
export OPENAI_API_KEY=sk-...

# Skips data download and container restart — tests against whatever is already running
make e2e-quick
```
---


> If you hit issues, see [DEBUGGING.md](DEBUGGING.md).

---

## Prerequisites

- Docker or Podman running (`docker ps` / `podman ps`)
- **Podman users:** `docker compose` is a Docker-only plugin. Use one of:
  - `alias docker=podman` — makes `docker compose` invoke `podman compose` (requires `podman-compose` installed)
  - Or use `podman compose` directly wherever `docker compose` appears in these docs
  - `make` targets auto-detect the runtime and don't need the alias

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
git clone git@github.com:IBM/vakra.git
cd vakra
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

## 3. Download Data and Start Containers

Data download is required for both routes below:

```bash
# Download benchmark data from HuggingFace (~30 GB)
# You will be prompted for a HuggingFace token
make download
```

> **Warning:** `make download` fetches ~30 GB of data. This will be reduced in a future release.

Then choose your route:

---

### Route A — Pull from Docker Hub (recommended for most users)

No build step required. Pulls the pre-built image and starts all containers:

```bash
make pull
make start
```

`make start` always pulls the latest image before starting, so you're guaranteed to be on the current published version.

---

### Route B — Build locally and use Docker Compose (for contributors and local changes)

Use this when you've made changes to server code or `docker/Dockerfile.unified` and want to test locally without pushing to Docker Hub first.

```bash
# Build the image from source
make build

# Start all four containers (uses the locally built image — no pull)
docker compose up -d
```

Wait ~60 seconds for the internal FastAPI services to initialize, then verify:

```bash
docker compose ps
docker compose logs -f   # Ctrl+C to exit
```

**Running e2e tests against locally built containers:**

```bash
export OPENAI_API_KEY=sk-...

# Skips data download and container restart — tests against whatever is already running
make e2e-quick
```

> `docker compose up -d` never pulls from Docker Hub. If you change server code, re-run `make build` before `docker compose up -d` to pick up the changes.

**MCP config files:**

The benchmark runner uses `benchmark/mcp_connection_config.yaml` by default. Two versions are available:

| Config file | Requires | Use when |
|-------------|----------|----------|
| `benchmark/mcp_connection_config.yaml` | New image (after `make build`) | Default — uses `/app/mcp_dispatch.py` for a single unified entrypoint |
| `benchmark/mcp_connection_config_legacy.yaml` | Any image version | You haven't rebuilt yet, or want direct server paths |

To use the legacy config:

```bash
python benchmark_runner.py --mcp-config benchmark/mcp_connection_config_legacy.yaml \
    --m3_capability_id 2 --domain address --provider openai
```

**Day-to-day after initial setup (either route):**

```bash
docker compose up -d    # restart containers without pulling or rebuilding
docker compose down     # stop and remove all containers
docker compose logs -f  # tail logs
```

---

## 4. Verify Docker Containers

Once setup completes, confirm four containers are running:

```bash
docker ps
```

You should see **4 containers** listed:

| Container | Purpose |
|-----------|---------|
| `capability_1_bi_apis_m3_environ` | Sel/Slot MCP server |
| `capability_2_dashboard_apis_m3_environ` | M3 REST MCP server |
| `capability_3_multihop_reasoning_m3_environ` | BPO MCP server + M3 REST API |
| `capability_4_multiturn_m3_environ` | M3 REST API + ChromaDB Retriever |

### Restarting a single container

If one container fails or becomes unhealthy while the others are fine, restart it individually rather than tearing everything down.

`make` targets work for both Docker and Podman. The raw `docker compose` commands below need `alias docker=podman` or `podman compose` on Podman systems.

```bash
# Via make — works for Docker and Podman without aliasing
make start-task1
make start-task2
make start-task3
make start-task5

# Or directly via docker compose (same effect)
docker compose up -d capability_1_bi_apis_m3_environ
docker compose up -d capability_4_multiturn_m3_environ

# Stop and remove a single container only
docker compose rm -sf capability_1_bi_apis_m3_environ

# Check logs for a specific container
docker logs capability_1_bi_apis_m3_environ --tail 50
docker compose logs -f capability_1_bi_apis_m3_environ
```

> `capability_4_multiturn_m3_environ` (ChromaDB) is the most likely to OOM on startup. If it exits immediately, confirm Docker Desktop memory is set to at least 8 GB (see [Prerequisites](#prerequisites)).

### Debugging containers

See [DEBUGGING.md](DEBUGGING.md) for container inspection commands, benchmark run logs, MCP server log capture, and jq recipes.

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
python benchmark_runner.py --m3_capability_id 1 --domain authors --max-samples-per-domain 1 --provider openai

# Task 2 — M3 REST SQL tools
python benchmark_runner.py --m3_capability_id 2 --provider openai --domain address
python benchmark_runner.py --m3_capability_id 2 --provider openai --domain airline
python benchmark_runner.py --m3_capability_id 2     --provider openai          # all domains

# Task 3 — BPO + M3 REST tools
python benchmark_runner.py --m3_capability_id 3 --domain airline --provider openai

# Task 5 — ChromaDB retriever
python benchmark_runner.py --m3_capability_id 5 --domain address --provider openai
```

**Common options:**

```bash
# Limit samples (good for quick tests)
python benchmark_runner.py --m3_capability_id 2 --domain hockey --max-samples-per-domain 5

# Choose provider and model
python benchmark_runner.py --m3_capability_id 2 --domain hockey --provider anthropic --model claude-sonnet-4-6
python benchmark_runner.py --m3_capability_id 2 --domain hockey --provider openai --model gpt-4o
python benchmark_runner.py --m3_capability_id 2 --domain hockey --provider ollama --model llama3.1:8b

# Run multiple tasks in parallel
python benchmark_runner.py --m3_capability_id 2 5 --domain address --parallel

# Just list available tools (no agent run)
python benchmark_runner.py --m3_capability_id 2 --domain hockey --list-tools

# Limit tools via embedding similarity (top-k)
python benchmark_runner.py --m3_capability_id 2 --domain hockey --top-k-tools 10
```

Results are saved to `output/task_{id}_{timestamp}/{domain}.json`.

---

## Integration & Unit Tests

### End-to-end tests

Two ways to run e2e tests depending on whether containers are already running:

---

**Option 1 — `make e2e-quick` / `make e2e-quick-<provider>` (containers already running)**

Use this after `make build && docker compose up -d` or `make start`. No data re-download, no container restart.

Each target validates the required credentials for its provider:

```bash
# OpenAI (default)
export OPENAI_API_KEY=sk-...
make e2e-quick

# RITS
export RITS_API_KEY=...
make e2e-quick-rits

# WatsonX
export WATSONX_APIKEY=...
export WATSONX_PROJECT_ID=...   # or WATSONX_SPACE_ID
make e2e-quick-watsonx

# LiteLLM proxy
export LITELLM_API_KEY=...
export LITELLM_BASE_URL=https://your-litellm-proxy
make e2e-quick-litellm

# Anthropic
export ANTHROPIC_API_KEY=...
make e2e-quick-anthropic
```

Model overrides: set `RITS_MODEL`, `WATSONX_MODEL`, `OPENAI_MODEL`, `LITELLM_MODEL`, or `ANTHROPIC_MODEL` before running.

---

**Option 2 — `make e2e` (full setup from scratch)**

Downloads data, starts containers, then runs tests. Requires both tokens.

```bash
export HF_TOKEN=hf_...
export OPENAI_API_KEY=sk-...
make e2e
```

Alternatively, use a `.env` file:

```bash
cp template_env .env
# edit .env: set HF_TOKEN and OPENAI_API_KEY
export $(grep -v '^#' .env | xargs)
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
docker run -d --name capability_1_bi_apis_m3_environ \
    -v "$(pwd)/data/databases:/app/db:ro" \
    -v "$(pwd)/apis/configs:/app/apis/configs:ro" \
    m3_environ

# Task 2 — M3 REST MCP server
docker run -d --name capability_2_dashboard_apis_m3_environ \
    -v "$(pwd)/data/databases:/app/db:ro" \
    -v "$(pwd)/apis/configs:/app/apis/configs:ro" \
    m3_environ

# Task 3 — BPO MCP server + M3 REST API
docker run -d --name capability_3_multihop_reasoning_m3_environ \
    -v "$(pwd)/data/databases:/app/db:ro" \
    -v "$(pwd)/apis/configs:/app/apis/configs:ro" \
    m3_environ

# Task 5 — ChromaDB Retriever (needs extra memory + retriever volumes)
docker run -d --name capability_4_multiturn_m3_environ \
    --memory=4g \
    -v "$(pwd)/data/databases:/app/db:ro" \
    -v "$(pwd)/apis/configs:/app/apis/configs:ro" \
    -v "$(pwd)/indexed_documents:/app/retrievers/chroma_data" \
    -v "$(pwd)/data/queries:/app/retrievers/queries:ro" \
    m3_environ
```

To stop and remove a single container:

```bash
docker rm -f capability_2_dashboard_apis_m3_environ
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

The smoke test validates the image in two tiers — sections 1, 3, and 4 run without any data; section 2 (FastAPI health) requires `data/databases` to be populated.

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
| 2. M3 REST FastAPI | Yes (`data/databases`) | `/openapi.json` responds and has at least one route |
| 3. BPO MCP handshake | No | `python /app/apis/bpo/mcp/server.py` returns a valid JSON-RPC response |
| 4. M3 REST MCP handshake | Yes (`data/databases`) | `python /app/m3-rest/mcp_server.py` returns a valid JSON-RPC response (requires FastAPI on port 8000) |

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
| `make start-task1` | Start `capability_1_bi_apis_m3_environ` only |
| `make start-task2` | Start `capability_2_dashboard_apis_m3_environ` only |
| `make start-task3` | Start `capability_3_multihop_reasoning_m3_environ` only |
| `make start-task5` | Start `capability_4_multiturn_m3_environ` only |
| `make stop` | Stop and remove all benchmark containers |
| `make clean` | Stop containers and remove the local `m3_environ` Docker image |
| `make e2e` | Run end-to-end benchmark tests (requires `HF_TOKEN` + `OPENAI_API_KEY`) |
| `make e2e-quick` | Run e2e tests against already-running containers — OpenAI provider (requires `OPENAI_API_KEY`) |
| `make e2e-quick-rits` | Same, using RITS provider (requires `RITS_API_KEY`) |
| `make e2e-quick-watsonx` | Same, using WatsonX provider (requires `WATSONX_APIKEY` + project/space ID) |
| `make e2e-quick-litellm` | Same, using LiteLLM proxy (requires `LITELLM_API_KEY` + `LITELLM_BASE_URL`) |
| `make e2e-quick-anthropic` | Same, using Anthropic provider (requires `ANTHROPIC_API_KEY`) |
| `make logs` | Last 20 log lines per container |

---

## Rebuilding the Docker Image

Run these commands from the **project root** whenever `docker/Dockerfile.unified`, `apis/bpo/`, or any other server code changes.

### 1. Build and test

```bash
make build      # docker build -t benchmark_environ -f docker/Dockerfile.unified .
make test       # smoke-test: file checks + M3 REST health + BPO/M3 MCP handshakes
```

After rebuilding, restart containers and run a live connection check:

```bash
make start      # docker compose up -d
make validate   # python benchmark/validate_clients.py — tests every MCP server
```

### 2. Restart containers to pick up the new image

```bash
make start
```

This starts all benchmark containers via `docker compose up -d`.

---

## Troubleshooting: Docker Hub Pull Rate Limit

If `make build` or `make pull` fails with:

```
toomanyrequests: You have reached your unauthenticated pull rate limit.
```

You're hitting Docker Hub's anonymous pull limit (100 pulls / 6 hours per IP). Fix with one of the options below.

### Option 1 — Log in to Docker Hub (recommended)

Authenticated accounts have a higher limit (200+ pulls / 6 hours). A free account is sufficient.

```bash
docker login   # or: podman login docker.io
```

Then retry `make build` or `make pull`.

### Option 2 — Use an alternative base image registry

Edit [docker/Dockerfile.unified](docker/Dockerfile.unified) line 26 to pull `python:3.11-slim` from a mirror instead of Docker Hub:

```dockerfile
# Microsoft Container Registry (no rate limit)
FROM mcr.microsoft.com/devcontainers/python:3.11

# GitHub Container Registry mirror (community-maintained)
FROM ghcr.io/tschm/python:3.11-slim
```

Then rebuild:

```bash
make build
```

### Option 3 — Use Podman with a configured registry mirror

If you're on a system with a local or corporate registry mirror, point Podman at it:

```bash
# Example: mirror configured at myregistry.example.com
DOCKER=podman make build
```

Or set a mirror in `/etc/containers/registries.conf`:

```toml
[[registry]]
prefix = "docker.io"
location = "myregistry.example.com"
```

---

## Optional: Expose FastAPI for Browser / Swagger UI

By default, the benchmark runner connects to containers via `docker exec` (stdio) and no ports are exposed. If you want to browse the Swagger UI or fetch the OpenAPI spec from your host, use the ports override file:

```bash
docker compose -f docker-compose.yml -f docker-compose.ports.yml up -d
```

| Capability | URL |
|------------|-----|
| Capability 1 | http://localhost:8010/docs |
| Capability 2 | http://localhost:8020/docs |
| Capability 3 | http://localhost:8030/docs |
| Capability 4 (REST) | http://localhost:8040/docs |
| Capability 4 (Retriever) | http://localhost:8041/docs |

You can also download the OpenAPI spec without port mapping using:

```bash
python examples/download_spec.py --capability-id 2
python examples/download_spec.py --capability-id 4 --port 8001   # retriever
```

---

## Optional: MCP Tools Explorer

A local web UI for browsing and invoking MCP tools interactively — without writing any code or running a benchmark.

### What it does

- **Capability + domain selector** — pick a capability (1–4) and a domain; the domain list is filtered to only the domains that exist for that capability
- **Tool browser** — searchable list of all tools the MCP server exposes for that capability/domain
- **Inspect** — click any tool to see its full description and parameter schema
- **Invoke** — fill in parameters via auto-generated form fields (or paste raw JSON) and call the tool live against the running container; see the result inline

### 1. Install dependencies

```bash
pip install fastapi uvicorn
```

These are lightweight and only needed for the explorer — they are not required for `benchmark_runner.py`.

### 2. Make sure containers are running

```bash
docker compose up -d   # or: make start
```

### 3. Start the explorer

Run from the **project root**:

```bash
uvicorn tools_explorer.app:app --reload --port 7860
```

Then open **http://localhost:7860** in your browser.

### Usage

1. Click a **Capability** card in the left sidebar — the domain dropdown updates automatically
2. Select a **Domain** and click **Load Tools**
3. Click any tool in the list to open its detail panel
4. Fill in parameters and click **Invoke** to call the tool against the live container
5. Use **{ } Raw JSON** to switch from form mode to a free-form JSON input

### Notes

- The explorer connects to containers the same way `benchmark_runner.py` does — via `docker exec` stdio, no port mapping required
- `--reload` restarts the server automatically when you edit `tools_explorer/app.py`; drop it for a stable session
- To run on a different port: `--port <port>`

---

## Optional: Phoenix / Arize Observability

[Phoenix](https://phoenix.arize.com) provides a local LLM tracing UI. It is off by default.

### 1. Start Phoenix alongside the benchmark containers

```bash
docker compose --profile phoenix up -d
```

Phoenix UI: http://localhost:6006
OTLP endpoint: http://localhost:6006/v1/traces

### 2. Install the Python packages

```bash
pip install arize-phoenix-otel openinference-instrumentation-langchain
# or via pyproject.toml extras:
pip install -e ".[phoenix]"
```

### 3. Run with tracing enabled

```bash
python benchmark_runner.py --capability_id 2 --domain hockey --phoenix
```

Additional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--phoenix` | off | Enable Phoenix tracing |
| `--phoenix-endpoint` | `http://localhost:6006/v1/traces` | OTLP HTTP endpoint |
| `--phoenix-project` | `enterprise-benchmark` | Project name in Phoenix UI |

If the Phoenix packages are not installed or Phoenix is not reachable, the benchmark continues without tracing (graceful degradation).
