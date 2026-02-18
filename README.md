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

The benchmark containers (especially Task 5 with ChromaDB) require a minimum of **8 GB of memory** allocated to your container runtime. With the default memory (often 2 GB), the retriever container will be OOM-killed on startup.

### Docker Desktop

1. Open **Docker Desktop** → **Settings** → **Resources**
2. Set **Memory** to at least **8 GB**
3. Click **Apply & Restart**

### Podman

```bash
podman machine stop
podman machine set --memory 8192
podman machine start
```

Verify with: `podman info | grep -i memTotal`

### Rancher Desktop

1. Open **Rancher Desktop** → **Preferences** → **Virtual Machine**
2. Set **Memory** to at least **8 GB** (8192 MiB)
3. Click **Apply** (the VM will restart)

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
3. Start 3 containers (`task_1_m3_environ`, `task_2_m3_environ`, `task_5_m3_environ`) — `task_5_m3_environ` is started with `--memory=4g` to support ChromaDB's memory requirements

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

## Benchmark Runner

`benchmark_runner.py` runs LLM agents against MCP tool servers and records trajectories + answers.

### Tasks

| Task | Container | Description |
|------|-----------|-------------|
| 2 | `fastapi-mcp-server` | M3 SQL tools |
| 5 | `retriever-mcp-server` | ChromaDB retriever |

### Quick Start

```bash
pip install mcp langchain-anthropic langgraph langchain-ollama

# Single task
python benchmark_runner.py --task_id 5 --run-agent --domain address

# Multiple tasks, sequential (default)
python benchmark_runner.py --task_id 2 5 --run-agent --domain address

# Multiple tasks, parallel
python benchmark_runner.py --task_id 2 5 --run-agent --domain address --parallel
```

### Common Options

```bash
# Limit samples per domain
python benchmark_runner.py --task_id 5 --run-agent --domain address --max-samples-per-domain 5

# Choose provider and model
python benchmark_runner.py --task_id 5 --run-agent --provider anthropic --model claude-sonnet-4-5-20250929

# List tools only (no agent run)
python benchmark_runner.py --task_id 5 --list-tools --domain address

# Tool shortlisting (top-k most relevant per query)
python benchmark_runner.py --task_id 2 --run-agent --domain hockey --top-k-tools 5
```

### All Options

| Flag | Description |
|------|-------------|
| `--task_id ID [ID ...]` | Task ID(s) to run (e.g. `2`, `5`, or `2 5`) |
| `--parallel` | Run multiple task_ids concurrently via `asyncio.gather` |
| `--run-agent` | Run the agent on benchmark queries |
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

A single Docker image that bundles all three MCP servers, so you only need one container instead of two or three.

| Server | Port / Protocol | Exec Command |
|--------|----------------|--------------|
| M3 REST (Task 2) | FastAPI :8000 | `python /app/m3-rest/mcp_server.py` |
| Retrievers (Task 5) | FastAPI :8001 | `python /app/retrievers/mcp_server.py` |
| Sel/Slot Tools | MCP stdio | `python -m apis.m3.python_tools.mcp` |

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

Verify that both FastAPI servers are running inside the container:

```bash
# M3 REST (port 8000)
curl http://localhost:8000/openapi.json | head -c 200

# Retrievers (port 8001)
curl http://localhost:8001/health
```

Test each MCP server via `docker exec`:

```bash
# Task 2 — M3 SQL tools
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0.1.0"}}}' \
  | docker exec -i -e MCP_DOMAIN=address m3_environ python /app/m3-rest/mcp_server.py

# Task 5 — Retriever tools
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0.1.0"}}}' \
  | docker exec -i -e MCP_DOMAIN=address m3_environ python /app/retrievers/mcp_server.py

# Sel/Slot tools
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0.1.0"}}}' \
  | docker exec -i -e MCP_DOMAIN=superhero m3_environ python -m apis.m3.python_tools.mcp
```

### Push to Docker Hub

```bash
docker tag m3_environ docker.io/amurthi44g1wd/m3_environ:latest
docker push docker.io/amurthi44g1wd/m3_environ:latest
```

### Benchmark Runner (Unified)

`benchmark_runner_single_docker_image.py` works like `benchmark_runner.py` but targets the single `m3_environ` container.

```bash
# Task 2 — M3 SQL tools
python benchmark_runner_single_docker_image.py --task_id 2 --run-agent --domain address

# Task 5 — Retriever tools
python benchmark_runner_single_docker_image.py --task_id 5 --run-agent --domain address

# Both tasks in parallel
python benchmark_runner_single_docker_image.py --task_id 2 5 --run-agent --domain address --parallel

# List tools only
python benchmark_runner_single_docker_image.py --task_id 5 --list-tools --domain address
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
