# ChromaDB Retriever Service

Semantic search (RAG) over 62 domain document collections. Exposes each domain as a vector search endpoint via FastAPI, wrapped by an MCP server for use with LLM agents.

```
LLM Agent
    |  MCP stdio
    v
MCP Server (mcp_server.py)
    |  HTTP :8001
    v
FastAPI Server (server.py)
    |
    v
ChromaDB (chroma_data/)  +  Granite Embeddings
```

---

## Quick Start (Docker — easiest)

Pull the image from Docker Hub and let it download data from HuggingFace automatically:

```bash
# Pull and start — data downloads from HuggingFace on first run
docker compose up -d

# Check progress (first run downloads ~2GB of ChromaDB data)
docker compose logs -f

# Verify it's healthy
docker compose ps
```

That's it. The container:
1. Downloads ChromaDB collections from `anupamamurthi/retriever-chroma-data` on HuggingFace
2. Starts FastAPI on port 8001
3. Starts MCP server over stdio

---

## Quick Start (Local — for development)

```bash
cd apis/retrievers
pip install -r requirements.txt
```

### Option A: Download pre-built data from HuggingFace

```bash
pip install huggingface_hub

# Download ChromaDB collections + query files
python hf_sync.py download --repo anupamamurthi/retriever-chroma-data

# Start the server
python run.py
```

### Option B: Index documents from scratch

```bash
# Index all domain documents into ChromaDB
python index_all_domains.py /path/to/docs_by_domains_20k

# This creates:
#   chroma_data/    - ChromaDB collections (one per domain)
#   queries/        - Sample query files for testing

# Start the server
python run.py
```

---

## Indexing

Index domain JSON files into ChromaDB collections. Each file becomes its own searchable collection.

```bash
# Index all domains
python index_all_domains.py /path/to/docs_by_domains_20k

# Index only the first 5 domains (for testing)
python index_all_domains.py /path/to/docs_by_domains_20k --max-domains 5
```

**Input format:** Each `{domain}_docs.json` file contains a list of document chunks:
```json
[
  {"doc_id": "12345", "text": "Title\nBody text of the document chunk..."},
  ...
]
```

**Output:**
- `chroma_data/` — ChromaDB persistent storage (one collection per domain)
- `queries/{domain}_queries.json` — Auto-generated sample queries for testing

**Embedding model:** IBM Granite (`ibm-granite/granite-embedding-english-r2`) via SentenceTransformers. Downloaded automatically on first run.

---

## Docker

### Using Docker Hub image (recommended)

```bash
# Pull and run
docker compose up -d

# First run downloads data from HuggingFace automatically
# Subsequent runs use cached data from Docker volumes

# Stop
docker compose down

# Stop and wipe data (forces re-download on next start)
docker compose down -v
```

### Building locally

```bash
# Build from source
docker compose build

# Start
docker compose up -d

# Check health
docker compose ps
docker inspect retriever-mcp-server --format '{{.State.Health.Status}}'

# View logs
docker compose logs -f

# Stop
docker compose down
```

### Pushing to Docker Hub

```bash
# Login to Docker Hub
docker login

# Build the image
docker build -t docker.io/amurthi44g1wd/retriever-mcp:latest .

# Push to Docker Hub
docker push docker.io/amurthi44g1wd/retriever-mcp:latest

# To use the pushed image instead of building locally,
# edit docker-compose.yml: comment out 'build: .' and
# uncomment 'image: docker.io/amurthi44g1wd/retriever-mcp:latest'
```

### Using local data instead of HuggingFace download

If you already have `chroma_data/` locally and don't want the container to download from HuggingFace:

```bash
# Override HF_REPO to empty to skip download, and bind-mount local data
HF_REPO="" docker compose run -d \
  -v ./chroma_data:/app/chroma_data \
  -v ./queries:/app/queries:ro \
  retriever-mcp
```

---

## API Endpoints

Once the server is running:

```bash
# List all available domains
curl http://localhost:8001/domains

# Query a domain (semantic search)
curl -X POST http://localhost:8001/address/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the CIA?", "n_results": 3}'

# Get chunk count for a domain
curl http://localhost:8001/address/collection/count

# Index new chunks
curl -X POST http://localhost:8001/address/chunks \
  -H "Content-Type: application/json" \
  -d '{"id": "doc1_0", "text": "Some document text..."}'

# Health check
curl http://localhost:8001/health
```

---

## Testing

Unified test runner with three modes:

```bash
# 1. Direct HTTP to FastAPI (default) — requires: python run.py
python test_queries.py address --max-queries 5
python test_queries.py address hockey

# 2. In-process MCP server — requires: python run.py
python test_queries.py --mode mcp address --max-queries 5

# 3. Docker container (MCP over stdio) — requires: docker compose up
python test_queries.py --mode docker address --max-queries 5

# Test ALL domains
python test_queries.py
python test_queries.py --mode docker
```

### Extra analysis

```bash
# Recall@K analysis (how many queries find the right topic at K=1,3,5,10)
python test_queries.py --recall-at-k address

# Negative query test (baseline distances for unrelated queries)
python test_queries.py --negative address

# Cross-domain leakage (query one domain's collection with another's queries)
python test_queries.py --cross-domain address hockey

# Combine multiple tests
python test_queries.py --recall-at-k --negative address --max-queries 10

# Adjust results per query
python test_queries.py address --n-results 10 --max-queries 5
```

---

## Hugging Face Sync

Push/pull the pre-built ChromaDB collections so other machines skip the indexing step.

### Setup

```bash
pip install huggingface_hub
```

1. Go to https://huggingface.co/settings/tokens
2. Create a new token with **Write** access (required for uploads; Read is enough for downloads)
3. Login:

```bash
huggingface-cli login
# Paste your token when prompted
```

### Upload

```bash
# Upload chroma_data/ and queries/ to HuggingFace
python hf_sync.py upload --repo anupamamurthi/retriever-chroma-data

# Upload a smaller variant
python hf_sync.py upload --repo anupamamurthi/retriever-chroma-5k --chroma-dir ./chroma_data_5k
```

### Download

```bash
# Download into the current directory (creates chroma_data/ and queries/)
python hf_sync.py download --repo anupamamurthi/retriever-chroma-data

# Download to a custom location
python hf_sync.py download --repo anupamamurthi/retriever-chroma-data --output-dir /data/retriever
```

---

## File Reference

| File | Purpose |
|------|---------|
| `chromadb_retriever.py` | Core retriever class + Granite embedding function |
| `server.py` | FastAPI server with per-domain endpoints |
| `mcp_server.py` | MCP server wrapping FastAPI (OpenAPI auto-discovery) |
| `run.py` | Starts FastAPI server on port 8001 |
| `index_all_domains.py` | Batch indexer: domain JSON files -> ChromaDB + query files |
| `test_queries.py` | Unified test runner (fastapi / mcp / docker modes) |
| `hf_sync.py` | Upload/download ChromaDB data to/from Hugging Face |
| `entrypoint.sh` | Container entrypoint: downloads data, starts servers |
| `docker-compose.yml` | Docker service definition |
| `Dockerfile` | Container image |
| `requirements.txt` | Python dependencies |
| `chroma_data/` | ChromaDB persistent storage (git-ignored) |
| `queries/` | Sample query files per domain |

---

## End-to-End Workflow

```bash
# === One-time setup (producer) ===

# 1. Index documents
python index_all_domains.py /path/to/docs_by_domains

# 2. Push data to HuggingFace
python hf_sync.py upload --repo anupamamurthi/retriever-chroma-data

# 3. Build and push Docker image
docker build -t docker.io/amurthi44g1wd/retriever-mcp:latest .
docker push docker.io/amurthi44g1wd/retriever-mcp:latest

# === Usage (consumer) ===

# 4. Start the service (downloads data from HF automatically)
docker compose up -d

# 5. Test it works
python test_queries.py address --max-queries 5
python test_queries.py --mode docker address --max-queries 5

# 6. Run the benchmark
cd ../..
python benchmark_runner.py --task_id 2 --run-agent --domain address
```
