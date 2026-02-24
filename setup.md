# Setup Guide

> **Note:** These steps create a clean environment so your existing branch and code are not affected.

## Prerequisites

- Docker or Podman running (`docker ps` / `podman ps`)
- If using Podman, alias it: `alias docker=podman`

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

---

## 3. Run Setup Script

```bash
python m3_setup.py
```

> **You will be prompted for a Hugging Face token.**
>
> **Warning:** This step downloads ~30 GB of data. This will be reduced in a future release.

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

### Start containers individually (If needed)

Use these if you need to start a specific container manually (run from project root):

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

## 5. Run a Benchmark

Install benchmark dependencies:

```bash
pip install -r requirements_benchmark.txt
# or manually:
pip install langchain-openai langchain mcp langchain-anthropic langgraph langchain-ollama
```

Set your API keys:

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
python benchmark_runner.py --m3_task_id 1 --domain superhero

# Task 2 — M3 REST SQL tools
python benchmark_runner.py --m3_task_id 2 --domain hockey
python benchmark_runner.py --m3_task_id 2 --domain airline
python benchmark_runner.py --m3_task_id 2              # all domains

# Task 3 — BPO + M3 REST tools
python benchmark_runner.py --m3_task_id 3 --domain airline

# Task 5 — ChromaDB retriever
python benchmark_runner.py --m3_task_id 5 --domain address
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

```bash
# Set required env vars
export HF_TOKEN=hf_...
export OPENAI_API_KEY=sk-...

# Run all e2e tests
python -m pytest tests/e2e/test_benchmark_e2e.py -v -s

# Or use a .env file
cp template_env .env
# edit .env: set HF_TOKEN and OPENAI_API_KEY
export $(grep -v '^#' .env | xargs)
python -m pytest tests/e2e/test_benchmark_e2e.py -v -s
```

Run a single task test:

```bash
python -m pytest tests/e2e/test_benchmark_e2e.py::TestBenchmarkE2E::test_task1_address -v -s
python -m pytest tests/e2e/test_benchmark_e2e.py::TestBenchmarkE2E::test_task2_address -v -s
python -m pytest tests/e2e/test_benchmark_e2e.py::TestBenchmarkE2E::test_task3_airline -v -s
python -m pytest tests/e2e/test_benchmark_e2e.py::TestBenchmarkE2E::test_task5_address -v -s
```

> Containers are started once and shared across all tests. Each task writes to its own isolated output directory.

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
| `make logs` | Last 20 log lines per container |

---

## Rebuilding and Pushing the Docker Image

Run these commands from the **project root** whenever `docker/Dockerfile.unified`, `apis/bpo/`, or any other server code changes.

### 1. One-command release (recommended)

```bash
make release
```

This runs `build → test → tag → push` in sequence and stops on the first failure.

### 2. Step by step

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

> **Note:** You must be logged in (`docker login docker.io`) with push access to `amurthi44g1wd`.

### 3. Restart containers to pick up the new image

```bash
make start
```

This stops/removes all existing benchmark containers, pulls the freshly pushed image, and restarts all four containers.
