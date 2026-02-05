# Examples Directory

This directory contains examples demonstrating how to use the M3Benchmark tool-calling agent framework with the slot-filling MCP server.

## Overview

The main example script (`demo.py`) demonstrates:
- **Agent-based query execution** using LangChain with MCP tools
- **Multiple LLM providers** (RITS/OpenAI-compatible API and Ollama)
- **Two connection modes** (stdio subprocess and WebSocket)
- **Handle-based result management** to efficiently process large datasets
- **127 test queries** from the superhero database

## Prerequisites

- **Python Version**: 3.11 or higher
- **LLM Provider** (choose one):
  - RITS API (OpenAI-compatible) with API key
  - Ollama (local, no API key needed)
- **Database**: SQLite database (superhero database included at `db/dev_databases/superhero/superhero.sqlite`)

## Installation

### Dependency Groups

The project uses pyproject.toml dependency groups for modular installation:

| Group | Purpose | When to Use |
|-------|---------|-------------|
| `core` | Basic data tools (pandas, numpy, sqlglot, pydantic, jinja2, dotenv) | Always installed |
| `mcp` | MCP server + LangChain (includes agents) | **Required for demo.py** |
| `agents` | LangChain only | Included in mcp |
| `rest` | FastAPI REST API server | Not needed for examples |
| `rag` | RAG capabilities | Optional |
| `dev` | Development tools (black, pytest, etc.) | Development only |
| `all` | All dependencies | Full installation |

### Installation Commands

**Recommended (minimal for examples):**
```bash
pip install -e ".[mcp]"
```

**Full installation:**
```bash
pip install -e ".[all]"
```

**Core only:**
```bash
pip install -e .
```

**Additional for Ollama:**
```bash
pip install langchain-ollama
```

### Verify Installation

```bash
# Check MCP server command is available
slot-filling-mcp --version

# Verify Python imports
python -c "from agents.tool_calling_agent import ToolCallingAgent; print('✓ Imports successful')"
```

## Database Setup

### Directory Structure

The project follows this database directory pattern:

```
db/
└── dev_databases/
    └── {database_name}/
        ├── {database_name}.sqlite       # The actual database file
        └── database_description/        # Schema documentation (optional)
```

### Default Database

- **Location**: `db/dev_databases/superhero/superhero.sqlite`
- **Status**: Included in repository
- **Pattern**: `db/dev_databases/{database_name}/{database_name}.sqlite`

### Using a Custom Database

Set the environment variable to point to your database:

```bash
export SUPERHERO_DB="/path/to/your/database.sqlite"
```

## Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `SUPERHERO_DB` | Path to SQLite database | No | `db/dev_databases/superhero/superhero.sqlite` |
| `RITS_API_KEY` | API key for RITS/OpenAI-compatible API | Yes (for --llm openai) | None |

### API Key Setup

**For RITS/OpenAI:**
```bash
export RITS_API_KEY="your-api-key-here"
```

**For Ollama:**
No API key needed (runs locally)

### Test Data

- **Location**: `apis/configs/mcp_init_mapping.json`
- **Contents**: 127 test instances with queries, expected answers, and initialization arguments

## Usage

### Command-Line Parameters

| Parameter | Type | Choices | Default | Description |
|-----------|------|---------|---------|-------------|
| `--llm` | string | `openai`, `ollama` | `openai` | LLM provider to use |
| `--model` | string | Any | `llama-3-3-70b-instruct` (openai)<br>`llama3.3` (ollama) | Model name |
| `--ollama-base-url` | string | URL | `http://localhost:11434` | Ollama server URL |
| `--mode` | string | `stdio`, `websocket` | `stdio` | Connection mode to MCP server |
| `--server-url` | string | WebSocket URL | None | WebSocket URL (required for websocket mode) |

### Connection Modes Explained

- **stdio** (default): Launches MCP server as subprocess with auto-start/stop. Easiest for local development.
- **websocket**: Connects to external MCP server. Requires `--server-url`. Useful for remote or persistent servers.

## Usage Examples

### Basic Usage (RITS + stdio)

```bash
export RITS_API_KEY="your-key-here"
cd /path/to/enterprise-benchmark
python examples/demo.py
```

### Ollama (Local Models)

**Start Ollama and run demo:**
```bash
# Terminal 1: Start Ollama server
ollama serve

# Terminal 2: Pull model and run demo
ollama pull llama3.3
python examples/demo.py --llm ollama
```

**With different model:**
```bash
python examples/demo.py --llm ollama --model llama3.1:8b
```

**With custom Ollama server URL:**
```bash
python examples/demo.py --llm ollama --ollama-base-url http://192.168.1.100:11434
```

### Custom OpenAI Models

```bash
export RITS_API_KEY="your-key-here"
python examples/demo.py --llm openai --model gpt-oss-120b
```

### WebSocket Mode (Two Terminals)

**Terminal 1 - Start MCP server:**
```bash
slot-filling-mcp \
  --db db/dev_databases/superhero/superhero.sqlite \
  --transport websocket \
  --port 8000 \
  --host 127.0.0.1
```

**Terminal 2 - Run demo:**
```bash
export RITS_API_KEY="your-key-here"
python examples/demo.py \
  --mode websocket \
  --server-url ws://localhost:8000/mcp
```

### Custom Database

```bash
export SUPERHERO_DB="/path/to/custom/database.sqlite"
export RITS_API_KEY="your-key-here"
python examples/demo.py
```

### Full Custom Configuration

```bash
export RITS_API_KEY="your-key-here"
export SUPERHERO_DB="db/dev_databases/superhero/superhero.sqlite"
python examples/demo.py \
  --llm openai \
  --model llama-3-3-70b-instruct \
  --mode stdio
```

## How It Works

The demo script follows this workflow:

1. **Setup**: Creates an LLM instance based on the provider choice (OpenAI-compatible or Ollama)

2. **Load Test Cases**: Reads 127 test queries from `apis/configs/mcp_init_mapping.json`

3. **Connect to MCP Server**:
   - **stdio mode**: Launches MCP server as subprocess
   - **websocket mode**: Connects to external MCP server

4. **Initialize Session**: Creates MCP client session and toolkit with available tools

5. **For Each Test Instance**:
   - Calls `get_data` tool with `tool_universe_id` to switch data universe
   - Loads pre-joined data for that specific query
   - Stores initial data in handle manager
   - Runs agent with the query
   - Agent uses tools (filter, sort, aggregate, transform, etc.) to answer

6. **Handle-Based Results**: Manages large datasets using handles (e.g., `filtered_data_1`, `sorted_superhero_2`) to avoid overwhelming LLM context with full data

### Tool Universe System

- Each test instance has a unique **tool universe ID** (e.g., `superhero_0`, `superhero_1`)
- The ID determines which tables to join and how
- Allows different queries against the same database with different starting configurations
- Provides flexibility for complex multi-table scenarios

## Troubleshooting

### ModuleNotFoundError: No module named 'mcp'

**Solution:**
```bash
pip install -e ".[mcp]"
```

### ValueError: You need to set the env var RITS_API_KEY

**Solution:**
```bash
export RITS_API_KEY="your-api-key-here"
```

### Ollama Connection Errors

**Solution:**
```bash
# Start Ollama server
ollama serve

# Verify Ollama is running
curl http://localhost:11434/api/tags

# Pull required model
ollama pull llama3.3
```

### Database Not Found

**Solution 1 - Run from project root:**
```bash
cd /path/to/enterprise-benchmark
python examples/demo.py
```

**Solution 2 - Use absolute path:**
```bash
export SUPERHERO_DB="/absolute/path/to/superhero.sqlite"
python examples/demo.py
```

### WebSocket Connection Failed

**Check these:**
- MCP server is running in websocket mode
- Port and host match between server and client
- Firewall settings allow the connection
- URL format is correct: `ws://host:port/mcp`

### Python Version Issues

**Check version:**
```bash
python --version  # Should be 3.11+
```

**Use specific version:**
```bash
python3.11 -m pip install -e ".[mcp]"
python3.11 examples/demo.py
```

### langchain-ollama Not Found

**Solution:**
```bash
pip install langchain-ollama
```

### Import Errors After Installation

**Solution - Reinstall in development mode:**
```bash
pip uninstall enterprise-benchmark invocable-api-hub
pip install -e ".[mcp]"
```

## Next Steps

- **Explore Tools**: Check `apis/m3/python_tools/tools/slot_filling_tools.py` for available data manipulation functions
- **Custom Queries**: Modify test queries in `apis/configs/mcp_init_mapping.json`
- **Agent Customization**: Review `agents/tool_calling_agent.py` for agent implementation details
- **Add Your Database**: Follow the database directory structure to add custom databases
- **REST API**: See `apis/m3/rest/README.md` for exposing databases via REST endpoints

## Additional Resources

- **Project Root**: See `/CLAUDE.md` for complete project documentation
- **Tool Registry**: `apis/m3/python_tools/tools/tool_registry.py`
- **MCP Server**: `apis/m3/python_tools/mcp/mcp_server.py`
- **Handle Manager**: `agents/result_handle_manager.py`
