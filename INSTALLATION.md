# Benchmark Runner - Installation Guide

## Quick Start

### 1. Install Dependencies

**Full installation (all providers):**
```bash
pip install -r requirements_benchmark.txt
```

**Minimal installation (single provider):**
```bash
# Edit requirements_benchmark_minimal.txt to uncomment your preferred provider
pip install -r requirements_benchmark_minimal.txt
```

### 2. Set Environment Variables

**For Anthropic Claude:**
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

**For OpenAI:**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

**For Ollama:**
No API key needed (local models)

**Set Task Directory:**
```bash
export TASK_2_DIR="/path/to/your/task_2/input/"
```

### 3. Verify Installation

Test that all imports work:
```bash
python -c "from agent_interface import create_agent; print('✓ Agent interface imported successfully')"
python -c "from mcp import ClientSession; print('✓ MCP imported successfully')"
python -c "from langgraph.prebuilt import create_react_agent; print('✓ LangGraph imported successfully')"
```

### 4. Run Benchmark

**List tools only (no agent):**
```bash
python benchmark_runner.py --task_id 2
```

**Run with agent (Ollama - default):**
```bash
python benchmark_runner.py --task_id 2 --run-agent
```

**Run with Anthropic:**
```bash
python benchmark_runner.py --task_id 2 --run-agent --provider anthropic
```

**Run specific domain:**
```bash
python benchmark_runner.py --task_id 2 --run-agent --domain hockey
```

## Dependency Overview

| Package | Purpose | Required By |
|---------|---------|-------------|
| `mcp` | Model Context Protocol client | benchmark_runner.py |
| `langchain-core` | Core LangChain abstractions | Both scripts |
| `langchain` | LangChain framework | Both scripts |
| `langgraph` | ReAct agent implementation | agent_interface.py |
| `langchain-anthropic` | Anthropic Claude integration | agent_interface.py (optional) |
| `langchain-openai` | OpenAI GPT integration | agent_interface.py (optional) |
| `langchain-ollama` | Ollama local models | agent_interface.py (optional) |

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`, ensure you've installed all requirements:
```bash
pip install -r requirements_benchmark.txt
```

### MCP Connection Issues

Ensure your container is running:
```bash
podman ps | grep fastapi-mcp-server
# or
docker ps | grep fastapi-mcp-server
```

### API Key Issues

Verify your environment variables:
```bash
echo $ANTHROPIC_API_KEY
echo $OPENAI_API_KEY
```

### Python Version

This project requires Python 3.10 or higher:
```bash
python --version  # Should be 3.10+
```

## System Requirements

- **Python:** 3.10 or higher
- **Container Runtime:** Docker or Podman
- **MCP Server:** Must be running in a container named `fastapi-mcp-server`
- **Memory:** At least 4GB RAM recommended for running LLMs
- **Disk Space:** Varies by LLM provider (Ollama models can be 4-8GB each)

## Advanced Configuration

### Using Different Container Names

```bash
python benchmark_runner.py --task_id 2 \
  --container-name my-custom-container \
  --container-runtime docker
```

### Limiting Samples

```bash
python benchmark_runner.py --task_id 2 \
  --run-agent \
  --max-samples-per-domain 5
```

### Custom Output Location

```bash
python benchmark_runner.py --task_id 2 \
  --run-agent \
  --output my_results.json
```

## Getting Help

Run with `--help` to see all options:
```bash
python benchmark_runner.py --help
```
