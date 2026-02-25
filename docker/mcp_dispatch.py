#!/usr/bin/env python3
"""
Single entrypoint for all M3 benchmark MCP servers.

Reads TASK_ID and exec's the appropriate server — zero proxy overhead
(same os.execv pattern used by apis/bpo/mcp/task3_router.py).

  TASK_ID=1  → python -m apis.m3.python_tools.mcp      (Slot-fill/Selection router)
  TASK_ID=2  → python /app/m3-rest/mcp_server.py        (M3 REST MCP)
  TASK_ID=3  → python /app/apis/bpo/mcp/task3_router.py (BPO ↔ M3 REST router)
  TASK_ID=5  → python /app/retrievers/task5_mcp_server.py (M3 REST + Retriever)

Usage inside container:
    docker exec -i -e TASK_ID=2 <container> python /app/mcp_dispatch.py
"""
import os
import sys

TASK_ID = os.environ.get("TASK_ID", "").strip()

_ROUTES = {
    "1": [sys.executable, "-m", "apis.m3.python_tools.mcp"],
    "2": [sys.executable, "/app/m3-rest/mcp_server.py"],
    "3": [sys.executable, "/app/apis/bpo/mcp/task3_router.py"],
    "5": [sys.executable, "/app/retrievers/task5_mcp_server.py"],
}

if TASK_ID not in _ROUTES:
    print(
        f"mcp_dispatch: TASK_ID must be one of {list(_ROUTES)}; got {TASK_ID!r}",
        file=sys.stderr,
    )
    sys.exit(1)

cmd = _ROUTES[TASK_ID]
os.execv(cmd[0], cmd)
