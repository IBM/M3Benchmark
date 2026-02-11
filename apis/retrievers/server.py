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
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from chromadb_retriever import ChromaDBRetriever, GraniteEmbeddingFunction

PERSIST_DIR = "./chroma_data"
PRELOAD = os.environ.get("PRELOAD_COLLECTIONS", "true").lower() == "true"


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
    n_results: int = 3


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


# --------------- Per-domain endpoints ---------------

@app.post("/{domain}/chunks", response_model=IndexResponse)
async def index_chunks(domain: str, request: IndexRequest):
    """Add chunks to a domain's collection."""
    retriever = _get_retriever(domain)
    prepared = _prepare_chunks(request.chunks)
    retriever.add_chunks(prepared)
    return IndexResponse(indexed=len(prepared))


@app.post("/{domain}/index-file", response_model=IndexResponse)
async def index_file(domain: str, request: IndexFileRequest):
    """Index all chunks from a JSON file into a domain's collection."""
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


@app.post("/{domain}/query", response_model=QueryResponse)
async def query(domain: str, request: QueryRequest):
    """Semantic search within a domain's collection."""
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


@app.get("/{domain}/chunks/{chunk_id}")
async def get_chunk(domain: str, chunk_id: str):
    """Get a specific chunk by ID from a domain's collection."""
    retriever = _get_retriever(domain)
    result = retriever.collection.get(ids=[chunk_id])
    if not result["ids"]:
        raise HTTPException(status_code=404, detail=f"Chunk '{chunk_id}' not found")
    return {
        "id": result["ids"][0],
        "text": result["documents"][0],
        "metadata": result["metadatas"][0],
    }


@app.delete("/{domain}/chunks/{chunk_id}")
async def delete_chunk(domain: str, chunk_id: str):
    """Delete a specific chunk from a domain's collection."""
    retriever = _get_retriever(domain)
    result = retriever.collection.get(ids=[chunk_id])
    if not result["ids"]:
        raise HTTPException(status_code=404, detail=f"Chunk '{chunk_id}' not found")
    retriever.collection.delete(ids=[chunk_id])
    return {"deleted": chunk_id}


@app.get("/{domain}/collection/count")
async def collection_count(domain: str):
    """Get the number of chunks in a domain's collection."""
    retriever = _get_retriever(domain)
    return {"domain": domain, "count": retriever.collection.count()}
