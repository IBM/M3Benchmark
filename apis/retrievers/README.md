# ChromaDB Retriever Service

Semantic search (RAG) over 62 domain document collections. Each domain is a ChromaDB vector collection served via FastAPI and wrapped by an MCP server for LLM agents.

```
LLM Agent ──MCP stdio──> MCP Server (mcp_server.py)
                              |  HTTP :8001
                              v
                         FastAPI (server.py)
                              |
                              v
                         ChromaDB (chroma_data/) + Granite Embeddings
```

---

## For Users (Consumer)

You need two things: **data from HuggingFace** and the **Docker container from Docker Hub**.

### 1. Download data

```bash
pip install huggingface_hub
cd apis/retrievers
python hf_sync.py download --repo anupamamurthi/retriever-chroma-data
```

Creates `chroma_data/` (~2 GB) and `queries/`.

### 2. Start the container

```bash
docker compose up -d
# or
docker run -d --name retriever-mcp-server -p 8001:8001 \
  -e PRELOAD_COLLECTIONS=true \
  -v ./chroma_data:/app/chroma_data -v ./queries:/app/queries:ro \
  -i -t docker.io/amurthi44g1wd/retriever-mcp:latest
```

### 3. Verify

```bash
curl http://localhost:8001/health         # ok
curl http://localhost:8001/domains        # list of 62 domains
```

### 4. Test

```bash
python test_queries.py address --max-queries 5                    # via FastAPI
python test_queries.py --mode docker address --max-queries 5      # via MCP in container
python test_queries.py --mode docker --max-queries 3              # all domains
```

### 5. Stop

```bash
docker compose down
# or
docker stop retriever-mcp-server && docker rm retriever-mcp-server
```

---

## For Maintainers (Producer)

One-time setup to create the data and publish everything.

### 1. Index documents into ChromaDB

```bash
cd apis/retrievers
pip install -r requirements.txt
python index_all_domains.py /path/to/docs_by_domains_20k
```

**Input:** `{domain}_docs.json` files — `[{"doc_id": "...", "text": "..."}]`
**Output:** `chroma_data/` (ChromaDB collections) + `queries/` (sample test queries)

### 2. Push data to HuggingFace

```bash
pip install huggingface_hub
huggingface-cli login
python hf_sync.py upload --repo anupamamurthi/retriever-chroma-data
```

### 3. Build and push Docker image

```bash
docker build -t docker.io/amurthi44g1wd/retriever-mcp:latest .
docker push docker.io/amurthi44g1wd/retriever-mcp:latest
```

The image contains code and dependencies only — data is bind-mounted at runtime.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/domains` | List available domains |
| POST | `/{domain}/query` | Semantic search (`{"question": "...", "n_results": 3}`) |
| POST | `/{domain}/chunks` | Index new chunks |
| POST | `/{domain}/index-file` | Index from a JSON file |
| GET | `/{domain}/chunks/{id}` | Get a specific chunk |
| DELETE | `/{domain}/chunks/{id}` | Delete a chunk |
| GET | `/{domain}/collection/count` | Chunk count |

---

## Testing

```bash
# Direct HTTP to FastAPI (requires: python run.py or container running)
python test_queries.py address --max-queries 5

# MCP in-process (requires: python run.py)
python test_queries.py --mode mcp address --max-queries 5

# MCP via Docker container (requires: docker compose up)
python test_queries.py --mode docker address --max-queries 5

# Extra analysis
python test_queries.py --recall-at-k address              # Recall@K
python test_queries.py --negative address                  # Negative queries
python test_queries.py --cross-domain address hockey       # Cross-domain leakage
python test_queries.py --list-tools address                # List MCP tools
```

---

## Local Development

```bash
cd apis/retrievers
pip install -r requirements.txt
python hf_sync.py download --repo anupamamurthi/retriever-chroma-data
python run.py   # starts FastAPI on :8001
```

---

## File Reference

| File | Purpose |
|------|---------|
| `chromadb_retriever.py` | Core retriever class + Granite embedding function |
| `server.py` | FastAPI server with per-domain endpoints |
| `mcp_server.py` | MCP server wrapping FastAPI (OpenAPI auto-discovery) |
| `run.py` | Starts FastAPI on port 8001 |
| `index_all_domains.py` | Batch indexer: domain docs -> ChromaDB + query files |
| `test_queries.py` | Unified test runner (fastapi / mcp / docker modes) |
| `hf_sync.py` | Upload/download ChromaDB data to/from HuggingFace |
| `entrypoint.sh` | Container entrypoint: starts FastAPI + MCP |
| `docker-compose.yml` | Service definition |
| `Dockerfile` | Container image |
| `requirements.txt` | Python dependencies |
| `chroma_data/` | ChromaDB storage (git-ignored, ~2 GB) |
| `queries/` | Sample query files per domain |
