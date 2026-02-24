# M3Benchmark

## 📊 Dataset

Please find more details about the dataset (download, schema, etc.) in [docs/dataset.md](docs/dataset.md) and APIs in [apis](apis).

## Download Data

- **Task #1:** [Simple](https://github.com/)
- **Task #2:** [MultiTurn](https://github.com/)

## Data Schema

| Field Name             | Type          | Description                                                                                                                                                           |
|------------------------|---------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `sample_id`       | string        | A unique identifier for each dialogue. |
| `domain`          | string        | Domain label for the dialogue. Possible values: "finance", "music", "movie3", "sports" ...... |
| `num_turns`       | int           | Number of turns in the dialogue |
| `turns`           | list          | List of dictionaries containing the question for each turn, answer, and API call|
| `tool_list`       | list          | List of available tools for answering questions within the dialogue.|
| `alt_ans`         | list          | Other valid gold standard answers to the question.|
| `scenarios`       | dict          | Description present in [docs/scenarios.md](docs/dataset.md)|

## 📏 Evaluation Metrics
M3 systems are evaluated using a scoring method that measures response quality to questions in the evaluation set. Responses are rated as perfect, acceptable, missing, or incorrect:

- Perfect: The response correctly answers the user question and contains no hallucinated content.

- Acceptable: The response provides a useful answer to the user question, but may contain minor errors that do not harm the usefulness of the answer.

- Missing: The answer does not provide the requested information. Such as “I don’t know”, “I’m sorry I can’t find …” or similar sentences without providing a concrete answer to the question.

- Incorrect: The response provides wrong or irrelevant information to answer the user question


Auto-evaluation: 
- Automatic evaluation employs rule-based matching and LLM assessment to check answer correctness. It will assign three scores: correct (1 point), missing (0 points), and incorrect (-1 point).


Please refer to [evaluation.py](evaluation.py) for more details on how the evaluation was implemented.

## Container Runtime Requirements

Task 5 (ChromaDB) requires at least **8 GB** allocated to your container runtime. See [SETUP.md — Container memory requirements](SETUP.md#container-memory-requirements) for instructions for Docker Desktop, Podman, and Rancher Desktop.

## Setup

```bash
# 1. Install with init dependencies
python -m venv .venv
source .venv/bin/activate

pip install -e ".[init]"


# 2. Run full setup (downloads ~35 GB of data)

python m3_setup.py


```

This will:
1. Download data from 4 HuggingFace repos into `data/`:
   - `anupamamurthi/db` — SQLite databases
   - `anupamamurthi/tasks` — benchmark input files (task 1, 2, 5)
   - `anupamamurthi/chroma_data` — ChromaDB vector collections
   - `anupamamurthi/queries` — retriever query configurations
2. Pull `docker.io/amurthi44g1wd/m3_environ:latest` and tag it as `m3_environ`
3. Start 4 containers — `task_5_m3_environ` is started with `--memory=4g` to support ChromaDB's memory requirements

| Container | Purpose |
|-----------|---------|
| `task_1_m3_environ` | Sel/Slot MCP server |
| `task_2_m3_environ` | M3 REST MCP server |
| `task_3_m3_environ` | BPO MCP server + M3 REST API |
| `task_5_m3_environ` | M3 REST API + ChromaDB Retriever |

You can also run individual steps:

```bash
python m3_setup.py --download-data      # just download data
python m3_setup.py --pull-image          # just pull the Docker image
python m3_setup.py --start-containers    # just start containers
python m3_setup.py --stop-containers     # stop and remove containers
```

## ✍️ How to run end-to-end evaluation?
1. **Install** specific dependencies
    ```bash
    pip install -r requirements.txt
    ```

2. **Configure environment variables** by copying the provided template and filling in your credentials:
    ```bash
    cp example_env .env
    ```
    Edit `.env` and replace the placeholder values (e.g. `your-rits-api-key-here`) with your actual API keys. See `example_env` for the full list of available options.

3. Test your model locally using `python evaluation.py`. This script will run answer generation and auto-evaluation.

## Validate MCP Servers

Before running the benchmark, verify that the MCP servers are healthy and contain the correct tools:

```bash
python benchmark/validate_clients.py
```

This script checks that each MCP server (Tasks 1, 2, 3 and 4) is running properly in its container and exposes the expected tools for all configured domains.

## Benchmark Runner

`benchmark_runner.py` runs LLM agents against MCP tool servers and records trajectories + answers.

### Tasks

| Task | Container | Description |
|------|-----------|-------------|
| 2 | `fastapi-mcp-server` | M3 SQL tools |
| 5 | `retriever-mcp-server` | ChromaDB retriever |

### Quick Start

```bash
pip install langchain-openai langchain mcp langchain-anthropic langgraph langchain-ollama sentence-transformers


# Single task
python benchmark_runner.py --m3_task_id 5 --domain address

# Multiple tasks, sequential (default)
python benchmark_runner.py --m3_task_id 2 5 --domain address

# Multiple tasks, parallel
python benchmark_runner.py --m3_task_id 2 5 --domain address --parallel
```

### Common Options

```bash
# Limit samples per domain
python benchmark_runner.py --m3_task_id 5 --domain address --max-samples-per-domain 5

# Choose provider and model
python benchmark_runner.py --m3_task_id 5 --provider anthropic --model claude-sonnet-4-5-20250929

# List tools only (no agent run)
python benchmark_runner.py --m3_task_id 5 --list-tools --domain address

# Tool shortlisting (top-k most relevant per query)
python benchmark_runner.py --m3_task_id 2 --domain hockey --top-k-tools 5
```

### All Options

| Flag | Description |
|------|-------------|
| `--m3_task_id ID [ID ...]` | M3 Task ID(s) to run (e.g. `2`, `5`, or `2 5`) |
| `--parallel` | Run multiple m3_task_ids concurrently via `asyncio.gather` |
| `--list-tools` | List available tools and exit |
| `--domain DOMAIN` | Filter to domain(s), repeatable |
| `--max-samples-per-domain N` | Cap queries per domain |
| `--provider` | `ollama` (default), `anthropic`, `openai`, `litellm`, `watsonx`, `rits` |
| `--model MODEL` | Model name (default: provider-specific) |
| `--top-k-tools K` | Keep top-K tools per query via embedding similarity |
| `--container-runtime` | `docker` or `podman` (default: auto-detect) |
| `--container-name` | Override container name from task config |
| `--output DIR` | Override output directory |

### Output

Results are saved to `output/task_{id}_{timestamp}/<domain>.json` in the current working directory.

```
output/
  task_2_feb_13_11_21am/
    address.json
    hockey.json
  task_5_feb_13_11_21am/
    address.json
    airline.json
```


## Unified Docker Image (`m3_environ`)

A single Docker image that bundles all MCP servers. Each task gets its own named container started from the same image.

| Server | Port / Protocol | Exec Command |
|--------|----------------|--------------|
| M3 REST (Tasks 2, 3) | FastAPI :8000 | `python /app/m3-rest/mcp_server.py` |
| BPO (Task 3) | MCP stdio | `python /app/apis/bpo/mcp/server.py` |
| Retrievers (Task 5) | FastAPI :8001 | `python /app/retrievers/mcp_server.py` |
| Sel/Slot Tools (Task 1) | MCP stdio | `python -m apis.m3.python_tools.mcp` |

### Option A: Pull from Docker Hub

```bash
docker pull docker.io/amurthi44g1wd/m3_environ:latest
docker tag docker.io/amurthi44g1wd/m3_environ:latest m3_environ
```

### Option B: Build locally

```bash
docker build -t m3_environ -f docker/Dockerfile.unified .
```

### Run

```bash
docker run -d --name m3_environ \
    -v "$(pwd)/apis/m3/rest/db:/app/db:ro" \
    -v "$(pwd)/apis/configs:/app/apis/configs:ro" \
    -v "$(pwd)/apis/retrievers/chroma_data:/app/retrievers/chroma_data" \
    -v "$(pwd)/apis/retrievers/queries:/app/retrievers/queries:ro" \
    -p 8000:8000 -p 8001:8001 \
    m3_environ
```

### Quick Test

Verify that the M3 REST FastAPI server is running inside a container:

```bash
# M3 REST (port 8000) — available in all containers
docker exec task_2_m3_environ curl -sf http://localhost:8000/openapi.json | head -c 200

# Retrievers (port 8001) — task_5 only
docker exec task_5_m3_environ curl -sf http://localhost:8001/health
```

Test each MCP server via `docker exec`:

```bash
MCP_INIT='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0.1.0"}}}'

# Task 2 — M3 SQL tools
echo "$MCP_INIT" | docker exec -i -e MCP_DOMAIN=address task_2_m3_environ python /app/m3-rest/mcp_server.py

# Task 3 — BPO tools (primary)
echo "$MCP_INIT" | docker exec -i task_3_m3_environ python /app/apis/bpo/mcp/server.py

# Task 3 — M3 REST tools (secondary, same container)
echo "$MCP_INIT" | docker exec -i -e MCP_DOMAIN=address task_3_m3_environ python /app/m3-rest/mcp_server.py

# Task 5 — Retriever tools
echo "$MCP_INIT" | docker exec -i -e MCP_DOMAIN=address task_5_m3_environ python /app/retrievers/mcp_server.py

# Task 1 — Sel/Slot tools
echo "$MCP_INIT" | docker exec -i -e MCP_DOMAIN=superhero task_1_m3_environ python -m apis.m3.python_tools.mcp
```

### Rebuilding and Pushing the Docker Image

Run these steps from the **project root** after any changes to `docker/Dockerfile.unified`, `apis/bpo/`, or other server code.

```bash
# 1. Build
docker build -t m3_environ -f docker/Dockerfile.unified .

# 2. Smoke-test — checks files, M3 REST health, and MCP handshakes for BPO + M3 REST
bash docker/smoke_test.sh
# Exit 0 = all good; exit 1 = fix issues before pushing

# 3. Tag and push
docker tag m3_environ docker.io/amurthi44g1wd/m3_environ:latest
docker push docker.io/amurthi44g1wd/m3_environ:latest

# 4. Restart all benchmark containers to use the new image
python m3_setup.py --start-containers
```

> **Note:** You must be logged in (`docker login docker.io`) with push access to `amurthi44g1wd`.

The smoke-test script (`docker/smoke_test.sh`) runs these checks against a temporary container:

| Check | What it verifies |
|-------|-----------------|
| File existence | All required `.py`, `.parquet`, and entrypoint files are in the image |
| M3 REST FastAPI | `/openapi.json` responds within 60 s and has at least one route |
| BPO MCP handshake | `python /app/apis/bpo/mcp/server.py` returns a valid JSON-RPC response |
| M3 REST MCP handshake | `python /app/m3-rest/mcp_server.py` returns a valid JSON-RPC response |

### Benchmark Runner (Unified)

`benchmark_runner_single_docker_image.py` works like `benchmark_runner.py` but targets the single `m3_environ` container.

```bash
# Task 2 — M3 SQL tools
python benchmark_runner_single_docker_image.py --m3_task_id 2 --domain address

# Task 5 — Retriever tools
python benchmark_runner_single_docker_image.py --m3_task_id 5 --domain address

# Both tasks in parallel
python benchmark_runner_single_docker_image.py --m3_task_id 2 5 --domain address --parallel

# List tools only
python benchmark_runner_single_docker_image.py --m3_task_id 5 --list-tools --domain address
```

### Demo (Unified)

`examples/demo_single_docker_image.py` runs the sel/slot MCP server against the unified container.

```bash
python examples/demo_single_docker_image.py --mode docker --domain superhero \
    --provider openai --model gpt-4o --max-samples-per-domain 1
```

## 🏁 Baselines
We include three baselines for demonstration purposes, and you can read more about them in [docs/baselines.md](docs/baselines.md).


## Citations


## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](LICENSE). This license permits sharing and adapting the work, provided it's not used for commercial purposes and appropriate credit is given. For a quick overview, visit [Creative Commons License](https://creativecommons.org/licenses/by-nc/4.0/).
