"""
FastAPI server exposing ChromaDB retrievers as REST APIs.
Supports multiple domain collections (e.g., /address/query, /hockey/query).

Run with:
    uvicorn server:app --reload
"""

import json
import os
from contextlib import asynccontextmanager

import chromadb
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel

from chromadb_retriever import ChromaDBRetriever, GraniteEmbeddingFunction

PERSIST_DIR = "./chroma_data"
PRELOAD = os.environ.get("PRELOAD_COLLECTIONS", "true").lower() == "true"

# Load domain descriptions for richer tool descriptions
_desc_path = os.path.join(os.path.dirname(__file__), "domain_description_dict.json")
try:
    with open(_desc_path) as _f:
        DOMAIN_DESCRIPTIONS: dict = json.load(_f)
except FileNotFoundError:
    DOMAIN_DESCRIPTIONS = {}


# --------------- Pydantic models ---------------

class Chunk(BaseModel):
    id: str
    text: str
    from_clapnq: bool = False


class IndexRequest(BaseModel):
    chunks: list[Chunk]


class IndexResponse(BaseModel):
    indexed: int


class IndexFileRequest(BaseModel):
    file_path: str


class QueryRequest(BaseModel):
    question: str
    n_results: int = 10


class QueryResult(BaseModel):
    id: str
    text: str
    distance: float
    metadata: dict


class QueryResponse(BaseModel):
    results: list[QueryResult]


# --------------- App setup ---------------

retrievers: dict[str, ChromaDBRetriever] = {}
embedding_fn: GraniteEmbeddingFunction = None


def _register_domain_routes(app: FastAPI, domain: str):
    """Register explicit routes for a domain so the OpenAPI spec has per-domain descriptions."""
    router = APIRouter(prefix=f"/{domain}", tags=[domain])

    @router.post("/chunks", response_model=IndexResponse,
                 summary=f"Index chunks into {domain}",
                 description=f"Add document chunks to the {domain} collection.")
    async def index_chunks(request: IndexRequest):
        retriever = _get_retriever(domain)
        prepared = _prepare_chunks(request.chunks)
        retriever.add_chunks(prepared)
        return IndexResponse(indexed=len(prepared))

    @router.post("/index-file", response_model=IndexResponse,
                 summary=f"Index a file into {domain}",
                 description=f"Index all chunks from a JSON file into the {domain} collection.")
    async def index_file(request: IndexFileRequest):
        try:
            with open(request.file_path) as f:
                raw_chunks = json.load(f)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

        retriever = _get_retriever(domain)
        chunks = [Chunk(**item) for item in raw_chunks]
        prepared = _prepare_chunks(chunks)
        retriever.add_chunks(prepared)
        return IndexResponse(indexed=len(prepared))

    domain_desc = DOMAIN_DESCRIPTIONS.get(domain, {}).get(
        "description", f"Retrieve document(s) that best match the query related to {domain}."
    )

    @router.post("/query", response_model=QueryResponse,
                 summary=f"Query {domain}",
                 description=domain_desc)
    async def query(request: QueryRequest):
        retriever = _get_retriever(domain)
        raw = retriever.query(request.question, n_results=request.n_results)

        results = []
        if raw["documents"] and raw["documents"][0]:
            for doc, dist, meta, cid in zip(
                raw["documents"][0],
                raw["distances"][0],
                raw["metadatas"][0],
                raw["ids"][0],
            ):
                results.append(QueryResult(id=cid, text=doc, distance=dist, metadata=meta))

        return QueryResponse(results=results)

    @router.get("/chunks/{chunk_id}",
                summary=f"Get chunk from {domain}",
                description=f"Get a specific chunk by ID from the {domain} collection.")
    async def get_chunk(chunk_id: str):
        retriever = _get_retriever(domain)
        result = retriever.collection.get(ids=[chunk_id])
        if not result["ids"]:
            raise HTTPException(status_code=404, detail=f"Chunk '{chunk_id}' not found")
        return {
            "id": result["ids"][0],
            "text": result["documents"][0],
            "metadata": result["metadatas"][0],
        }

    @router.delete("/chunks/{chunk_id}",
                   summary=f"Delete chunk from {domain}",
                   description=f"Delete a specific chunk from the {domain} collection.")
    async def delete_chunk(chunk_id: str):
        retriever = _get_retriever(domain)
        result = retriever.collection.get(ids=[chunk_id])
        if not result["ids"]:
            raise HTTPException(status_code=404, detail=f"Chunk '{chunk_id}' not found")
        retriever.collection.delete(ids=[chunk_id])
        return {"deleted": chunk_id}

    @router.get("/collection/count",
                summary=f"Count chunks in {domain}",
                description=f"Get the number of chunks in the {domain} collection.")
    async def collection_count():
        retriever = _get_retriever(domain)
        return {"domain": domain, "count": retriever.collection.count()}

    app.include_router(router)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_fn
    # Load the embedding model once, shared across all retrievers
    embedding_fn = GraniteEmbeddingFunction()

    if PRELOAD:
        client = chromadb.PersistentClient(path=PERSIST_DIR)
        collections = client.list_collections()
        print(f"Pre-loading {len(collections)} collections...")
        for col in collections:
            _get_retriever(col.name)
        print(f"Loaded: {', '.join(retrievers.keys())}")
    yield


app = FastAPI(
    title="ChromaDB Retriever API",
    description="Index and retrieve text chunks using ChromaDB with Granite embeddings. "
                "Each domain has its own collection (e.g., /address/query, /hockey/query).",
    lifespan=lifespan,
)

# Discover collections at module level and register per-domain routes
# so they appear in the OpenAPI spec before the app starts serving.
_client = chromadb.PersistentClient(path=PERSIST_DIR)
_collections = _client.list_collections()
print(f"Registering {len(_collections)} domain(s): {', '.join(sorted(c.name for c in _collections))}")
for _col in _collections:
    _register_domain_routes(app, _col.name)
del _client, _collections, _col


# --------------- Helpers ---------------

def _get_retriever(domain: str) -> ChromaDBRetriever:
    """Get or create a retriever for a domain, reusing the shared embedding model."""
    if domain not in retrievers:
        retriever = ChromaDBRetriever.__new__(ChromaDBRetriever)
        retriever.embedding_fn = embedding_fn
        retriever.client = chromadb.PersistentClient(path=PERSIST_DIR)
        retriever.collection = retriever.client.get_or_create_collection(
            name=domain,
            embedding_function=embedding_fn,
        )
        retrievers[domain] = retriever
    return retrievers[domain]


def _prepare_chunks(chunks: list[Chunk]) -> list[dict]:
    """Convert Chunk models to the format expected by ChromaDBRetriever."""
    prepared = []
    for i, chunk in enumerate(chunks):
        prepared.append({
            "id": f"{chunk.id}_{i}",
            "text": chunk.text,
            "metadata": {
                "doc_id": chunk.id,
                "from_clapnq": chunk.from_clapnq,
            },
        })
    return prepared


# --------------- Global endpoints ---------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/domains")
async def list_domains():
    """List all available domain collections."""
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collections = client.list_collections()
    return {
        "domains": [c.name for c in collections],
        "count": len(collections),
    }


