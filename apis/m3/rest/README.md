# FastAPI + MCP Server

A FastAPI server exposing 80+ database APIs with 9,800+ endpoints, wrapped by an MCP (Model Context Protocol) server for LLM tool calling.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Docker Container                     │
│  ┌─────────────────┐       ┌─────────────────────────┐  │
│  │   FastAPI       │       │      MCP Server         │  │
│  │   (port 8000)   │◄─────►│   (stdio / port 8001)   │  │
│  │                 │       │                         │  │
│  │  80+ API modules│       │  Converts OpenAPI spec  │  │
│  │  9,800+ routes  │       │  to MCP tools           │  │
│  └─────────────────┘       └─────────────────────────┘  │
│           │                                              │
│           ▼                                              │
│  ┌─────────────────┐                                    │
│  │   SQLite DBs    │                                    │
│  │   (./db/)       │                                    │
│  └─────────────────┘                                    │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Build and start the container
docker-compose up -d

# Check logs
docker logs fastapi-mcp-server -f

# Test the API
curl http://localhost:8000/docs
```

### Option 2: Local Development

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run FastAPI server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## Setup Details

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for containerized deployment)
- SQLite databases in `./db/` directory

### Project Structure

```
rest/
├── app.py                 # FastAPI application entry point
├── server/                # API modules (80 files, one per database)
│   ├── __init__.py
│   ├── address.py
│   ├── airline.py
│   ├── hockey.py
│   └── ...
├── db/                    # SQLite databases
│   ├── address/
│   ├── airline/
│   └── ...
├── mcp_server.py          # MCP server wrapper
├── docker-compose.yml     # Docker orchestration
├── Dockerfile.mcp         # Container build instructions
└── requirements.txt       # Python dependencies
```

## Running with Docker

### Start Services

```bash
# Build and start (foreground)
docker-compose up

# Build and start (background)
docker-compose up -d

# Force rebuild
docker-compose build --no-cache
docker-compose up -d
```

### Using Podman

```bash
# Remove existing container
podman rm -f fastapi-mcp-server

# Build and start
podman-compose build
podman-compose up -d
```

### Check Status

```bash
# View running containers
docker ps

# View logs
docker logs fastapi-mcp-server -f

# Stop services
docker-compose down
```

### Ports

| Port | Service |
|------|---------|
| 8000 | FastAPI server (REST API + Swagger docs) |
| 8001 | MCP server (for external MCP connections) |

## Running Locally

### Start FastAPI Server

```bash
# Activate virtual environment
source .venv/bin/activate

# Start server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Or specify a different port
uvicorn app:app --port 8003
```

### Start MCP Server (separate terminal)

```bash
# Set FastAPI URL
export FASTAPI_BASE_URL=http://localhost:8000

# Run MCP server
python mcp_server.py
```

## API Documentation

Once running, access the interactive API docs:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Using with LangChain Agent

Run the example agent that connects to the MCP server in Docker:

```bash
# Set your API key
export ANTHROPIC_API_KEY=your-key-here
# Or use Ollama (free, local)
export USE_OLLAMA=true

# Run the agent
python examples/langchain_agent_docker_remote.py
```

The agent will:
1. Connect to the MCP server in the Docker container
2. Discover all available tools (API endpoints)
3. Use the LLM to answer queries by calling the appropriate APIs

## Health Check

```bash
# Check if server is running
curl http://localhost:8000/health

# Check metrics (Prometheus)
curl http://localhost:8000/metrics
```

## Troubleshooting

### Container won't start

```bash
# Check logs for errors
docker logs fastapi-mcp-server

# Rebuild from scratch
docker-compose down
docker-compose build --no-cache
docker-compose up
```

### Database connection errors

Ensure the `db/` directory exists and contains the SQLite databases:

```bash
ls -la db/
```

### Import errors

Make sure `server/__init__.py` exists:

```bash
touch server/__init__.py
```

### Port already in use

```bash
# Find process using port
lsof -i :8000

# Kill it or use a different port
uvicorn app:app --port 8003
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FASTAPI_BASE_URL` | `http://localhost:8000` | URL for MCP server to reach FastAPI |
| `ANTHROPIC_API_KEY` | - | API key for Claude models |
| `OPENAI_API_KEY` | - | API key for OpenAI models |
| `USE_OLLAMA` | - | Set to use local Ollama models |

## Operation ID Metadata

After updating operation IDs, mapping files are available in `server/metadata/`:

- Individual mappings: `{module}_operation_id_map.json`
- Combined mappings: `all_operation_id_mappings.json`

These map old auto-generated operation IDs to the new function-name-based IDs.
