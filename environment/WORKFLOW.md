# End-to-End Workflow

This document describes the full lifecycle: from raw data to running benchmarks. There are two roles — **Producer** (one-time setup) and **Consumer** (day-to-day usage).

```
 PRODUCER (one-time)                          CONSUMER (usage)
 ==================                           ================

 Raw data                                     Source repo
   |                                             |
   v                                             v
 1. Index documents ──> ChromaDB data         4. Build image from source
   |                                             |
   v                                             v
 2. Push data ──────> HuggingFace             5. Download data from HF
   |                                             |
   v                                             v
 3. Build + push ───> HuggingFace             6. docker compose up -d
                                                 |
                                                 v
                                              7. Verify + test
                                                 |
                                                 v
                                              8. Run benchmark
```

---

## Two Services

| | M3 Tools (port 8000) | Retrievers (port 8001) |
|---|---|---|
| **What** | SQL query endpoints over 80 SQLite databases | Semantic search over 62 ChromaDB collections |
| **Tools per domain** | 6–441 (9,891 total) | 6 per domain (query, chunks, count, etc.) |
| **Data source** | [BIRD-Bench](https://bird-bench.github.io/) train + dev SQLite files | Domain document JSONs, indexed with Granite embeddings |
| **Data location** | `apis/m3/rest/db/` | `apis/retrievers/chroma_data/` |
| **Docker image** | `benchmark_environ` (built from source) | `benchmark_environ` (built from source) |
| **Container name** | `capability_2_dashboard_apis` | `capability_4_multiturn` |

Both services follow the same pattern: **FastAPI** serves REST endpoints, **MCP Server** wraps them via OpenAPI auto-discovery, and `MCP_DOMAIN` filters tools per domain.

---

## Producer Workflow (one-time setup)

### Step 1 — Index documents into ChromaDB

Start with raw domain document files (`{domain}_docs.json`), each containing a list of text chunks.

```bash
cd apis/retrievers
pip install -r requirements.txt

# Index all 62 domains (~2 hours, creates chroma_data/ and queries/)
python index_all_domains.py /path/to/docs_by_domains_20k

# Or index a subset for testing
python index_all_domains.py /path/to/docs_by_domains_20k --max-domains 5
```

**Input:** `{domain}_docs.json` files with format `[{"doc_id": "...", "text": "..."}, ...]`
**Output:**
- `chroma_data/` — ChromaDB persistent storage (one collection per domain, ~2GB)
- `queries/{domain}_queries.json` — Auto-generated sample queries for testing

### Step 2 — Push data to HuggingFace

So other machines can skip the indexing step.

```bash
pip install huggingface_hub
huggingface-cli login   # needs a Write token from https://huggingface.co/settings/tokens

cd apis/retrievers
python hf_sync.py upload --repo anupamamurthi/retriever-chroma-data
```

This uploads `chroma_data/` and `queries/` to the HF dataset repo.

### Step 3 — Build Docker image

```bash
make build
```

The container image contains the server code and dependencies but NOT the data. Data is bind-mounted at runtime.

---

## Consumer Workflow (day-to-day usage)

### Step 4 — Download data

```bash
pip install huggingface_hub

# Retrievers: download ChromaDB collections + query files
cd apis/retrievers
python hf_sync.py download --repo anupamamurthi/retriever-chroma-data
# Creates: chroma_data/ (~2GB) and queries/

# M3 Tools: download SQLite databases from BIRD-Bench
# See apis/m3/rest/README.md for download instructions
# Place databases under apis/m3/rest/db/{domain}/{domain}.sqlite
```

### Step 5 — Start the containers

```bash
docker compose up -d
```

### Step 6 — Verify both services

```bash
# M3 Tools
curl http://localhost:8000/health
curl http://localhost:8000/docs          # Swagger UI

# Retrievers
curl http://localhost:8001/health
curl http://localhost:8001/domains
```

### Step 7 — Test

```bash
cd apis/retrievers

# List tools for a domain
python test_queries.py --list-tools address                          # retriever endpoints
python test_queries.py --list-tools --base-url http://localhost:8000 address  # M3 tools

# Run retriever test queries
python test_queries.py address --max-queries 5                      # via FastAPI
python test_queries.py --mode docker address --max-queries 5        # via MCP in container

# Test all 62 retriever domains
python test_queries.py --mode docker --max-queries 3
```

### Step 8 — Run the benchmark

```bash
cd /path/to/enterprise-benchmark-1

export TASK_2_DIR=/path/to/task_2_data

# Run with Ollama (local)
python benchmark_runner.py --capability_id 2 --run-agent \
  --provider ollama --model qwen2.5-coder:7b \
  --domain address --max-samples-per-domain 5

# Run with Anthropic
python benchmark_runner.py --capability_id 2 --run-agent \
  --provider anthropic \
  --domain address

# Run with OpenAI
python benchmark_runner.py --capability_id 2 --run-agent \
  --provider openai \
  --domain hockey --max-samples-per-domain 10

# Run all domains
python benchmark_runner.py --capability_id 2 --run-agent
```

---

## Stopping

```bash
docker compose down
```

---

## How the benchmark works

```
benchmark_runner.py
       |
       |  Reads {domain}.json files from TASK_2_DIR
       |  Each file contains queries + expected tools
       |
       +---> For each domain:
       |
       |     1. Spawns MCP server(s) inside container(s)
       |        docker exec -i -e MCP_DOMAIN={domain} fastapi-mcp-server python mcp_server.py
       |        docker exec -i -e MCP_DOMAIN={domain} retriever-mcp-server python mcp_server.py
       |
       |     2. MCP server connects to FastAPI, discovers OpenAPI spec,
       |        filters to domain endpoints, exposes as MCP tools
       |
       |     3. LLM agent receives the tools and the user query
       |        - Decides which tools to call
       |        - Calls tools via MCP (stdin/stdout)
       |        - MCP server proxies to FastAPI
       |        - FastAPI queries SQLite / ChromaDB
       |        - Results flow back to agent
       |
       |     4. Agent formulates final answer
       |
       +---> Results saved to benchmark_output_{timestamp}/
                - {domain}_benchmark_output.json per domain
                - summary.json with overall scores
```

---

## Directory structure

```
enterprise-benchmark-1/
  benchmark_runner.py             # Main benchmark orchestrator
  evaluation.py                   # Scoring (perfect/acceptable/missing/incorrect)
  agents/                         # LLM agent implementations
  apis/
    ARCHITECTURE.md               # Architecture diagrams
    WORKFLOW.md                   # This file
    m3/rest/                      # M3 Tools service
      app.py                      #   FastAPI app (80 domain routers)
      mcp_server.py               #   MCP wrapper
      server/                     #   80 domain modules (address.py, hockey.py, ...)
      db/                         #   SQLite databases (one per domain, git-ignored)
      Dockerfile
      docker-compose.yml
    retrievers/                   # Retrievers service
      server.py                   #   FastAPI app (per-domain endpoints)
      mcp_server.py               #   MCP wrapper
      chromadb_retriever.py       #   Core retriever + Granite embeddings
      index_all_domains.py        #   Batch indexer (docs -> ChromaDB + queries)
      test_queries.py             #   Unified test runner (fastapi/mcp/docker + --list-tools)
      hf_sync.py                  #   HuggingFace upload/download helper
      chroma_data/                #   ChromaDB storage (git-ignored, ~2GB)
      queries/                    #   Sample query files per domain
      Dockerfile
      docker-compose.yml
```
