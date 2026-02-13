#!/bin/bash
set -e

# Check that chroma_data exists
if [ ! -d "/app/chroma_data" ] || [ -z "$(ls -A /app/chroma_data 2>/dev/null)" ]; then
    echo "ERROR: No chroma_data/ found."
    echo "Download it first:"
    echo "  python hf_sync.py download --repo anupamamurthi/retriever-chroma-data"
    echo "Then mount it into the container."
    exit 1
fi

# Start FastAPI server
echo "Starting FastAPI server on port 8001..."
uvicorn server:app --host 0.0.0.0 --port 8001 &

echo "Waiting for FastAPI server..."
until curl -sf http://localhost:8001/health > /dev/null 2>&1; do
    sleep 1
done
echo "FastAPI is up."

# Start MCP server (or run custom command)
if [ $# -gt 0 ]; then
    exec "$@"
else
    echo "Starting MCP server..."
    exec python mcp_server.py
fi
