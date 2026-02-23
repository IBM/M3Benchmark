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

Run a sample benchmark:

```bash
python benchmark_runner.py \
  --m3_task_id 2 \
  --domain airline \
  --max-samples-per-domain 1 \
  --provider openai

# Task 3 (BPO + M3 REST tools)
python benchmark_runner.py \
  --m3_task_id 3 \
  --domain airline \
  --max-samples-per-domain 1 \
  --provider openai
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
