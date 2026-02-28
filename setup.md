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
pip install -e ".[init]"
pip install -r requirements_benchmark.txt

# 2. Set required tokens
export HF_TOKEN=hf_...
export OPENAI_API_KEY=sk-...

# 3. Download benchmark data (~30 GB)
make download

# 4. Stop any existing containers, build the image, and start all 4 containers
docker compose down    # or: podman compose down
make build
docker compose up -d   # or: podman compose up -d

# 5. Wait ~60 s for internal services to initialize, then verify
docker compose ps      # or: podman compose ps

# 6. Run a single-sample smoke test (Task 1, authors domain)
python benchmark_runner.py --m3_task_id 1 --domain california_schools --max-samples-per-domain 1 --provider openai

# Results land in output/task_1_<timestamp>/authors.json
```

After making changes to server code or `docker/Dockerfile.unified`, re-run `make build` then `docker compose up -d` (or `podman compose up -d`) to pick them up.

```bash
export OPENAI_API_KEY=sk-...

# Skips data download and container restart — tests against whatever is already running
make e2e-quick
```
---

### Option 2 — Pull pre-built image from Docker Hub

Use this if you just want to run the benchmark without building anything.

```bash
# 1. Python environment
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[init]"
pip install -r requirements_benchmark.txt

# 2. Set required tokens
export HF_TOKEN=hf_...
export OPENAI_API_KEY=sk-...

# 3. Download benchmark data (~30 GB), pull the image, and start all 4 containers
#    (make start stops existing containers before starting fresh ones)
make download
make pull
make start

# 4. Run a single-sample smoke test (Task 1, authors domain)
python benchmark_runner.py --m3_task_id 1 --domain authors --max-samples-per-domain 1 --provider openai

# Results land in output/task_1_<timestamp>/authors.json
```


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
    --m3_task_id 2 --domain address --provider openai
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
| `task_1_m3_environ` | Sel/Slot MCP server |
| `task_2_m3_environ` | M3 REST MCP server |
| `task_3_m3_environ` | BPO MCP server + M3 REST API |
| `task_5_m3_environ` | M3 REST API + ChromaDB Retriever |

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
docker compose up -d task_1_m3_environ
docker compose up -d task_5_m3_environ

# Stop and remove a single container only
docker compose rm -sf task_1_m3_environ

# Check logs for a specific container
docker logs task_1_m3_environ --tail 50
docker compose logs -f task_1_m3_environ
```

> `task_5_m3_environ` (ChromaDB) is the most likely to OOM on startup. If it exits immediately, confirm Docker Desktop memory is set to at least 8 GB (see [Prerequisites](#prerequisites)).

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

### End-to-end tests

Two ways to run e2e tests depending on whether containers are already running:

---

**Option 1 — `make e2e-quick` (containers already running)**

Use this after `make build && docker compose up -d` or `make start`. No data re-download, no container restart.

```bash
export OPENAI_API_KEY=sk-...
make e2e-quick
```

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
| `make start-task1` | Start `task_1_m3_environ` only |
| `make start-task2` | Start `task_2_m3_environ` only |
| `make start-task3` | Start `task_3_m3_environ` only |
| `make start-task5` | Start `task_5_m3_environ` only |
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
