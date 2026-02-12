#!/bin/bash
set -e

# ---- Download data from HuggingFace if not already present ----
if [ -n "$HF_REPO" ] && [ ! -d "/app/chroma_data" ] || [ -z "$(ls -A /app/chroma_data 2>/dev/null)" ]; then
    echo "Downloading ChromaDB data from HuggingFace: $HF_REPO"
    python hf_sync.py download --repo "$HF_REPO" --output-dir /app
    echo "Download complete."
elif [ -d "/app/chroma_data" ] && [ -n "$(ls -A /app/chroma_data 2>/dev/null)" ]; then
    echo "ChromaDB data already present, skipping download."
else
    echo "WARNING: No HF_REPO set and no chroma_data/ found."
    echo "Set HF_REPO env var or mount chroma_data/ volume."
fi

# ---- Start FastAPI server ----
echo "Starting FastAPI server on port 8001..."
uvicorn server:app --host 0.0.0.0 --port 8001 &

echo "Waiting for FastAPI server..."
until curl -sf http://localhost:8001/health > /dev/null 2>&1; do
    sleep 1
done
echo "FastAPI is up."

# ---- Start MCP server (or run custom command) ----
if [ $# -gt 0 ]; then
    exec "$@"
else
    echo "Starting MCP server..."
    exec python mcp_server.py
fi
