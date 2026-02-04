# FastAPI + MCP Server

A FastAPI server exposing 80 database APIs with 9,800+ endpoints, wrapped by MCP (Model Context Protocol) for LLM tool calling. The MCP server supports domain filtering so agents can connect to only the endpoints they need.

## Architecture

```
                          ┌─────────────────────────────────────────────────────────┐
                          │                  Docker Container                       │
                          │                                                         │
                          │   ┌─────────────────────────────────────────────────┐   │
                          │   │            FastAPI  (port 8000)                 │   │
                          │   │                                                 │   │
                          │   │   /v1/hockey/*          122 endpoints           │   │
                          │   │   /v1/movie/*           148 endpoints           │   │
                          │   │   /v1/superhero/*        95 endpoints           │   │
                          │   │   /v1/financial/*       130 endpoints           │   │
                          │   │   ...                                           │   │
                          │   │   80 domains        9,800+ total endpoints      │   │
                          │   │                                                 │   │
                          │   │   SQLite databases (./db/)                      │   │
                          │   │   OpenAPI spec at /openapi.json                 │   │
                          │   │   Health check at /health                       │   │
                          │   └──────────────────────┬──────────────────────────┘   │
                          │                          │                              │
                          │                 http://localhost:8000                    │
                          │                          │                              │
                          │   ┌──────────────────────┴──────────────────────────┐   │
                          │   │            MCP Server (stdio)                   │   │
                          │   │                                                 │   │
                          │   │   1. Fetches /openapi.json from FastAPI         │   │
                          │   │   2. Filters paths by MCP_DOMAINS env var      │   │
                          │   │   3. Converts matching endpoints → MCP tools   │   │
                          │   │   4. Serves tools over stdio (stdin/stdout)    │   │
                          │   └──────────────────────┬──────────────────────────┘   │
                          │                          │                              │
                          └──────────────────────────│──────────────────────────────┘
                                                     │ stdio
                                                     │
                          ┌──────────────────────────┴──────────────────────────┐
                          │              LangChain ReAct Agent                  │
                          │                                                     │
                          │   Spawns MCP server via:                            │
                          │     docker exec -i -e MCP_DOMAINS=hockey \         │
                          │       fastapi-mcp-server python mcp_server.py      │
                          │                                                     │
                          │   OR locally:                                       │
                          │     MCP_DOMAINS=hockey python mcp_server.py        │
                          │                                                     │
                          │   LLM (Claude / Ollama / OpenAI) picks tools       │
                          │   from the filtered set and calls them via MCP     │
                          └─────────────────────────────────────────────────────┘
```

### How domain filtering works

```
  Agent sets MCP_DOMAINS="hockey"
       │
       ▼
  MCP Server starts
       │
       ├── Fetches /openapi.json (all 9,800+ endpoints)
       │
       ├── Filters: keep only paths starting with /v1/hockey/
       │
       ├── Converts 122 matching endpoints → 122 MCP tools
       │
       └── Agent sees only hockey tools
```

### Local development

```
  ┌────────────────────────────────────────────────────────────┐
  │                FastAPI  (port 8000)                        │
  │         uvicorn app:app --host 0.0.0.0 --port 8000        │
  └────────────────────────────┬───────────────────────────────┘
                               │
                      http://localhost:8000
                               │
                  ┌────────────┴────────────┐
                  │   MCP Server (stdio)    │
                  │                         │
                  │   MCP_DOMAINS=hockey    │
                  │   → only hockey tools   │
                  └────────────┬────────────┘
                               │ stdio
                               ▼
                          LLM Agent
```

The agent spawns `MCP_DOMAINS="hockey" python mcp_server.py` which
connects to FastAPI on port 8000 and exposes only the matching tools.

## All 80 Domains

| # | Domain | # | Domain |
|---|--------|---|--------|
| 1 | address | 41 | mondial_geo |
| 2 | airline | 42 | movie |
| 3 | app_store | 43 | movie_3 |
| 4 | authors | 44 | movie_platform |
| 5 | beer_factory | 45 | movies_4 |
| 6 | bike_share_1 | 46 | movielens |
| 7 | book_publishing_company | 47 | music_platform_2 |
| 8 | books | 48 | music_tracker |
| 9 | california_schools | 49 | olympics |
| 10 | car_retails | 50 | professional_basketball |
| 11 | card_games | 51 | public_review_platform |
| 12 | cars | 52 | regional_sales |
| 13 | chicago_crime | 53 | restaurant |
| 14 | citeseer | 54 | retail_complains |
| 15 | codebase_comments | 55 | retail_world |
| 16 | codebase_community | 56 | retails |
| 17 | coinmarketcap | 57 | sales |
| 18 | college_completion | 58 | sales_in_weather |
| 19 | computer_student | 59 | shakespeare |
| 20 | cookbook | 60 | shipping |
| 21 | craftbeer | 61 | shooting |
| 22 | cs_semester | 62 | simpson_episodes |
| 23 | debit_card_specializing | 63 | soccer_2016 |
| 24 | disney | 64 | social_media |
| 25 | donor | 65 | software_company |
| 26 | european_football_1 | 66 | student_club |
| 27 | european_football_2 | 67 | student_loan |
| 28 | financial | 68 | superhero |
| 29 | food_inspection | 69 | superstore |
| 30 | food_inspection_2 | 70 | synthea |
| 31 | formula_1 | 71 | talkingdata |
| 32 | genes | 72 | thrombosis_prediction |
| 33 | hockey | 73 | toxicology |
| 34 | human_resources | 74 | trains |
| 35 | ice_hockey_draft | 75 | university |
| 36 | image_and_language | 76 | video_games |
| 37 | language_corpus | 77 | works_cycles |
| 38 | law_episode | 78 | world |
| 39 | legislator | 79 | world_development_indicators |
| 40 | mental_health_survey | | |

## Push to Docker Hub

### Step 1 — Log in to Docker Hub

```bash
docker login docker.io -u <your-dockerhub-username>
```

### Step 2 — Build the image

```bash
cd /path/to/rest
docker build -t docker.io/<your-dockerhub-username>/m3:latest .
```

### Step 3 — Push the image

```bash
docker push docker.io/<your-dockerhub-username>/m3:latest
```

### Step 4 — Pull and run on another machine

```bash
docker pull docker.io/<your-dockerhub-username>/m3:latest

docker run -d -p 8000:8000 \
  -v ./db:/app/db:ro \
  --name fastapi-mcp-server \
  docker.io/<your-dockerhub-username>/m3:latest \
  bash -c "uvicorn app:app --host 0.0.0.0 --port 8000"
```

---

## Quick Start: Docker

### Step 1 — Build and start the container

```bash
cd /path/to/rest
docker-compose up -d --build
```

This builds the image and starts a single container running both FastAPI (port 8000) and the MCP server (stdio).

### Step 2 — Verify FastAPI is running

```bash
# Health check
curl http://localhost:8000/health

# Browse the API docs
open http://localhost:8000/docs
```

### Step 3 — Connect an agent to the MCP server

The MCP server inside the container communicates over stdio. To scope it to a specific domain, set `MCP_DOMAINS` in `docker-compose.yml`:

```yaml
environment:
  - FASTAPI_BASE_URL=http://localhost:8000
  - MCP_DOMAINS=airline        # only expose /v1/airline/* tools
```

Then restart:

```bash
docker-compose up -d
```

To expose all 9,800+ tools (all 80 domains), remove the `MCP_DOMAINS` line.

### Step 4 — View logs / stop

```bash
docker logs fastapi-mcp-server -f   # tail logs
docker-compose down                  # stop and remove container
```

---

## Quick Start: Local Development

### Step 1 — Install dependencies

```bash
cd /path/to/rest
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2 — Start the FastAPI server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Verify it's running:

```bash
curl http://localhost:8000/health
open http://localhost:8000/docs
```

### Step 3 — Start MCP server(s)

**Option A: Single MCP server (one domain)**

```bash
# In a new terminal (with .venv activated)
MCP_DOMAINS="hockey" python mcp_server.py
```

**Option B: Single MCP server (multiple domains)**

```bash
MCP_DOMAINS="hockey,movie,financial" python mcp_server.py
```

**Option C: Single MCP server (all domains)**

```bash
python mcp_server.py
```

**Option D: All 80 MCP servers at once**

```bash
./start_all_mcp_servers.sh
```

Each domain gets its own process on ports 8001–8080.

### Step 4 — Manage local MCP servers

```bash
./start_all_mcp_servers.sh status                 # Check which are running
./start_all_mcp_servers.sh start hockey movie      # Start specific ones
./start_all_mcp_servers.sh stop                    # Stop all
./start_all_mcp_servers.sh list                    # Show domain → port mapping
./start_all_mcp_servers.sh logs hockey             # Tail logs for a domain
```

### Step 5 — Run the test suite

```bash
# With FastAPI running on port 8000
python test_mcp_filtering.py
```

This tests no-filter, single-domain, multi-domain, and per-domain tool counts.

## Project Structure

```
rest/
├── app.py                      # FastAPI application (includes all 80 routers)
├── mcp_server.py               # MCP server wrapper (supports domain filtering)
├── Dockerfile                  # Container image for FastAPI + MCP
├── docker-compose.yml          # Single container: FastAPI + MCP
├── generate_compose.py         # Script to generate multi-container compose (optional)
├── start_all_mcp_servers.sh    # Script to run 80 MCP servers locally
├── test_mcp_filtering.py       # Test script for domain filtering
├── server/                     # API modules (80 files, one per database)
│   ├── __init__.py
│   ├── address.py              # /v1/address/* endpoints
│   ├── airline.py              # /v1/airline/* endpoints
│   ├── hockey.py               # /v1/hockey/* endpoints
│   └── ...                     # 77 more domain modules
├── db/                         # SQLite databases (see "Database Setup" below)
│   ├── address/
│   │   └── address.sqlite
│   ├── airline/
│   │   └── airline.sqlite
│   ├── hockey/
│   │   └── hockey.sqlite
│   └── ...                     # 80 domain directories, one .sqlite each
├── examples/                   # LangChain agent examples
│   ├── langchain_agent_local.py
│   └── langchain_agent_docker_remote.py
└── requirements.txt
```

## Database Setup

The SQLite databases must be placed under `m3/rest/db/`. The data comes from the [BIRD-Bench](https://bird-bench.github.io/) benchmark (train and dev splits).

### Download

1. Download the **train** and **dev** database splits from BIRD-Bench https://bird-bench.github.io/ 
2. Place each database in its own subdirectory under `db/`:

```
rest/db/
├── address/
│   └── address.sqlite
├── airline/
│   └── airline.sqlite
├── hockey/
│   └── hockey.sqlite
├── movie/
│   └── movie.sqlite
├── superhero/
│   └── superhero.sqlite
└── ...                    (80 directories total)
```

Each directory name must match the domain name used in the API routes (e.g., `hockey` for `/v1/hockey/*`). Each directory contains a single `.sqlite` file.

The `db/` directory is mounted read-only into Docker via `docker-compose.yml`:

```yaml
volumes:
  - ./db:/app/db:ro
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FASTAPI_BASE_URL` | `http://localhost:8000` | URL for MCP server to reach FastAPI |
| `MCP_SERVER_NAME` | `fastapi-mcp-wrapper` | Name for this MCP server instance |
| `MCP_DOMAINS` | (none = all) | Comma-separated domains to include (e.g., `hockey,movie`) |
| `ANTHROPIC_API_KEY` | - | API key for Claude models |
| `OPENAI_API_KEY` | - | API key for OpenAI models |
| `USE_OLLAMA` | - | Set to use local Ollama models |

## API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Using with LangChain Agent

```bash
export ANTHROPIC_API_KEY=your-key-here
python examples/langchain_agent_docker_remote.py
```

## Health Check

```bash
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

## Quick Test

Run the Docker image from Docker Hub and test with a LangChain agent.

### Step 1 — Free port 8000 (if in use) and copy db files from Bird Bench

```bash
lsof -i :8000
kill -9 <PID>
```

Download train and dev set from https://bird-bench.github.io/ - Unzip train and dev and inside the unzipped version, you will find a databases.zip file. That needs to be unzipped too. Copy the contents of databases (from both train and dev and place them under /db)

Copy the databases from the train and dev set into m3/rest/db. The folder structure should look like the below

<img width="307" alt="image" src="https://github.ibm.com/user-attachments/assets/47d8f55b-0d00-406f-bde9-7fa494817c34" />


### Step 2 — Start the container

```
podman init
podman machine init
podman machine start
podman ps
alias docker=podman
```

```bash
cd ./apis/m3/rest/
docker rm -f fastapi-mcp-server
docker run -d -p 8000:8000 \
  -v ./db:/app/db:ro \
  --name fastapi-mcp-server \
  docker.io/amurthi44g1wd/m3:latest \
  bash -c "uvicorn app:app --host 0.0.0.0 --port 8000"
```

### Step 3 — Verify it's running

```bash
docker ps
curl http://localhost:8000/docs
```
### Step 4 — Run the benchmark runner which points to a LangGraph agent

Download the “benchmark files” from https://ibm.ent.box.com/folder/364205927270 

export TASK_2_DIR=<point to the above directory in your disk>
Example: TASK_2_DIR=“/Users/anu/Documents/GitHub/routing/EnterpriseBenchmark/task_2" which holds two files



```
cd .
```

Make sure you have ollama set up in your machine.

```bash
python benchmark_runner.py --task_id 2 --run-agent --provider ollama --model qwen2.5-coder:7b --max-samples-per-domain 5 --domain airline
```

One could also use OPEN AI or ANTHROPIC models - refer to the USAGE block in benchmark_runner.py

### Step 5 — Run a different agent

```
cd ./apis/m3/rest
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
export OPENAI_API_KEY=your-key-here
MCP_DOMAINS="airline" python examples/langchain_agent_docker_remote.py
```


### Step 5 — Steps to start MCP servers alone

```
docker exec -i  -e MCP_DOMAINS=hockey fastapi-mcp-server python mcp_server.py
```

Note, it is key to pass the MCP_DOMAINS variable when starting the mcp server. MCP_DOMAINS points to the db - address vs bike_share vs airline etc

---

## Troubleshooting

### Container won't start

```bash
docker logs fastapi-mcp-server
docker-compose down && docker-compose build --no-cache && docker-compose up
```

### Port in use

```bash
lsof -i :8000
```

### Database errors

```bash
ls -la db/
```

### Rebuild after code changes

```bash
docker-compose build && docker-compose up -d
```
