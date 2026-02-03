# FastAPI + MCP Tools + MCP Retriever — Multi-Container Architecture

A two-container system where an LLM agent dynamically loads **tools** (API endpoints) and **retrievers** (RAG/document search) per domain via MCP, all controlled by a single domain parameter.

## Architecture

```
 ┌─────────────────────────────────────────────────────────────────────────────────────────┐
 │                                  LangChain ReAct Agent                                  │
 │                                                                                         │
 │   agent = create_react_agent(llm, tools=[...mcp_tools, ...mcp_retrievers])              │
 │                                                                                         │
 │   On startup for domain="hockey":                                                       │
 │     1. Spawns Tools MCP server      → gets 122 API tools                                │
 │     2. Spawns Retriever MCP server  → gets hockey retriever tool                        │
 │     3. Merges both tool sets into a single agent                                        │
 │                                                                                         │
 │   LLM decides whether to call an API tool or search documents                           │
 └────────────┬──────────────────────────────────────────────┬─────────────────────────────┘
              │ stdio                                        │ stdio
              │                                              │
 ┌────────────┴───────────────────────┐   ┌──────────────────┴────────────────────────────┐
 │  Docker Container: "tools"         │   │  Docker Container: "retriever"                │
 │                                    │   │                                               │
 │  ┌──────────────────────────────┐  │   │  ┌─────────────────────────────────────────┐  │
 │  │   FastAPI  (port 8000)       │  │   │  │  Retriever Service  (port 9000)         │  │
 │  │                              │  │   │  │                                         │  │
 │  │  /v1/hockey/*   122 eps     │  │   │  │  Chroma / FAISS vector stores           │  │
 │  │  /v1/movie/*    148 eps     │  │   │  │  per domain (./vectordb/)               │  │
 │  │  /v1/superhero/* 95 eps     │  │   │  │                                         │  │
 │  │  ...                         │  │   │  │  POST /retrieve                         │  │
 │  │  80 domains, 9,800+ eps     │  │   │  │    { domain, query, top_k }             │  │
 │  │                              │  │   │  │    → ranked doc chunks                  │  │
 │  │  SQLite databases (./db/)    │  │   │  │                                         │  │
 │  └──────────────┬───────────────┘  │   │  │  GET /health                            │  │
 │                 │                   │   │  └──────────────┬──────────────────────────┘  │
 │    http://localhost:8000            │   │                 │                              │
 │                 │                   │   │    http://localhost:9000                       │
 │  ┌──────────────┴───────────────┐  │   │                 │                              │
 │  │  MCP Server (stdio)          │  │   │  ┌──────────────┴──────────────────────────┐  │
 │  │                              │  │   │  │  MCP Server (stdio)                     │  │
 │  │  Filters OpenAPI by          │  │   │  │                                         │  │
 │  │  MCP_DOMAINS env var         │  │   │  │  Filters retriever tools by             │  │
 │  │  → API call tools            │  │   │  │  MCP_DOMAINS env var                    │  │
 │  └──────────────────────────────┘  │   │  │  → search_<domain>_docs tool            │  │
 │                                    │   │  └──────────────────────────────────────────┘  │
 └────────────────────────────────────┘   └───────────────────────────────────────────────┘
```

## How domain filtering works (both containers)

```
  Agent sets DOMAIN="hockey"
       │
       ├──► Tools MCP Server                     ├──► Retriever MCP Server
       │     │                                    │     │
       │     ├─ Fetches /openapi.json             │     ├─ Loads hockey vector store
       │     ├─ Filters: /v1/hockey/* only        │     ├─ Exposes search_hockey_docs tool
       │     ├─ 122 API tools                     │     │   (query → top-k doc chunks)
       │     └─ Agent sees API tools              │     └─ Agent sees retriever tool
       │                                          │
       └──────────── Agent merges both sets ──────┘
                     → 122 API tools + 1 retriever tool
```

## Data flow example

```
  User: "Who holds the record for most goals and what does the rulebook say about overtime?"
       │
       ▼
  LangChain Agent
       │
       ├─ Calls get_hockey_players_stats (API tool from Tools container)
       │   → FastAPI → SQLite → structured data
       │
       ├─ Calls search_hockey_docs (retriever tool from Retriever container)
       │   → Chroma vector search → relevant rulebook passages
       │
       └─ LLM synthesizes both into final answer
```

## Docker Compose — Two Containers

```yaml
# docker-compose.yml
services:
  tools:
    build:
      context: .
      dockerfile: Dockerfile.tools
    container_name: fastapi-mcp-tools
    ports:
      - "8000:8000"
    volumes:
      - ./db:/app/db:ro
    environment:
      - FASTAPI_BASE_URL=http://localhost:8000
      - MCP_DOMAINS=hockey          # optional: filter to one domain

  retriever:
    build:
      context: .
      dockerfile: Dockerfile.retriever
    container_name: mcp-retriever
    ports:
      - "9000:9000"
    volumes:
      - ./vectordb:/app/vectordb:ro
    environment:
      - RETRIEVER_BASE_URL=http://localhost:9000
      - MCP_DOMAINS=hockey          # same domain filter
```

## Project Structure

```
rest/
├── app.py                          # FastAPI application (80 API routers)
├── mcp_server.py                   # Tools MCP server (domain-filtered API tools)
├── retriever/
│   ├── retriever_app.py            # FastAPI retriever service (vector search)
│   ├── mcp_retriever_server.py     # Retriever MCP server (domain-filtered retriever tools)
│   ├── ingest.py                   # Script to build vector stores from documents
│   └── requirements.txt
├── vectordb/                       # Vector stores (one per domain)
│   ├── hockey/
│   ├── movie/
│   └── ...                         # 80 domain directories
├── Dockerfile.tools                # Container image for FastAPI + Tools MCP
├── Dockerfile.retriever            # Container image for Retriever + Retriever MCP
├── docker-compose.yml              # Both containers
├── server/                         # API modules (80 files)
│   ├── hockey.py
│   └── ...
├── db/                             # SQLite databases
│   ├── hockey/hockey.sqlite
│   └── ...
├── examples/
│   ├── langchain_agent_local.py
│   ├── langchain_agent_docker_remote.py
│   └── langchain_agent_dual.py     # Agent that loads BOTH tools + retriever
└── requirements.txt
```

## Quick Start

### Step 1 — Build and start both containers

```bash
docker-compose up -d --build
```

### Step 2 — Verify both services

```bash
curl http://localhost:8000/health    # Tools container
curl http://localhost:9000/health    # Retriever container
```

### Step 3 — Run the dual agent

```bash
export ANTHROPIC_API_KEY=your-key-here

# Agent loads tools + retriever for hockey domain
python examples/langchain_agent_dual.py --domain hockey
```

The agent spawns two MCP server processes:

```bash
# Tools MCP (inside tools container)
docker exec -i -e MCP_DOMAINS=hockey fastapi-mcp-tools python mcp_server.py

# Retriever MCP (inside retriever container)
docker exec -i -e MCP_DOMAINS=hockey mcp-retriever python mcp_retriever_server.py
```

Both tool sets are merged into a single LangChain agent.

### Step 4 — Switch domains

```bash
# Same containers, different domain
python examples/langchain_agent_dual.py --domain movie
python examples/langchain_agent_dual.py --domain financial
```

No container restart needed — the domain filter is passed at MCP server spawn time.

## Environment Variables

| Variable | Default | Container | Description |
|----------|---------|-----------|-------------|
| `FASTAPI_BASE_URL` | `http://localhost:8000` | tools | URL for Tools MCP to reach FastAPI |
| `RETRIEVER_BASE_URL` | `http://localhost:9000` | retriever | URL for Retriever MCP to reach retriever service |
| `MCP_DOMAINS` | (none = all) | both | Comma-separated domains to filter |
| `MCP_SERVER_NAME` | `fastapi-mcp-wrapper` | tools | MCP server instance name |
| `ANTHROPIC_API_KEY` | — | agent | API key for Claude |
| `OPENAI_API_KEY` | — | agent | API key for OpenAI |

## All 80 Domains

Both containers support the same 80 domains. Each domain has:
- **Tools container**: REST API endpoints backed by SQLite
- **Retriever container**: Vector store for document/knowledge search

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
