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

### Output

Results are saved to `<task_input_dir>/../output/<domain>.json`.


## 🏁 Baselines
We include three baselines for demonstration purposes, and you can read more about them in [docs/baselines.md](docs/baselines.md).


## Citations


## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](LICENSE). This license permits sharing and adapting the work, provided it's not used for commercial purposes and appropriate credit is given. For a quick overview, visit [Creative Commons License](https://creativecommons.org/licenses/by-nc/4.0/).
