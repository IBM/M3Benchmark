#!/usr/bin/env python3
"""
Task 3 MCP Router
=================

Routes to the correct MCP server based on MCP_DOMAIN by exec'ing into it.
The calling process is fully replaced, so the MCP client communicates directly
with the target server over stdio — no proxy overhead.

  MCP_DOMAIN=bpo     → /app/apis/bpo/mcp/server.py  (BPO FastMCP server)
  MCP_DOMAIN=<other> → /app/m3-rest/mcp_server.py   (M3 REST MCP server)

Usage (inside capability_3_multihop_reasoning container):
  docker exec -i -e MCP_DOMAIN=bpo     capability_3_multihop_reasoning python /app/apis/bpo/mcp/task3_router.py
  docker exec -i -e MCP_DOMAIN=airline capability_3_multihop_reasoning python /app/apis/bpo/mcp/task3_router.py
"""
import os
import sys

BPO_SERVER = "/app/apis/bpo/mcp/server.py"
M3_REST_SERVER = "/app/m3-rest/mcp_server.py"

domain = os.environ.get("MCP_DOMAIN", "")
target = BPO_SERVER if domain == "bpo" else M3_REST_SERVER

os.execv(sys.executable, [sys.executable, target])
