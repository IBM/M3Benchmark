#!/usr/bin/env python3
"""
Test script for MCP server domain filtering.

Usage:
1. Start the FastAPI server:
   uvicorn app:app --port 8000

2. Run this test:
   python test_mcp_filtering.py
"""
import asyncio
from mcp_server import FastAPIMCPServer


async def test_filtering():
    url = "http://localhost:8000"

    print("=" * 60)
    print("MCP Server Domain Filtering Test")
    print("=" * 60)

    # Test 1: No filtering (all endpoints)
    print("\n[1] No filtering — all endpoints")
    print("-" * 40)
    server_all = FastAPIMCPServer(fastapi_base_url=url)
    try:
        await server_all.initialize()
        print(f"  Total tools: {len(server_all.tools_cache)}")
    except Exception as e:
        print(f"  ERROR: {e}")
        print("  Make sure FastAPI is running on http://localhost:8000")
        return

    # Collect unique domains from paths
    domains = set()
    for tool in server_all.tools_cache:
        path = tool._metadata["path"]
        parts = path.split("/")
        if len(parts) >= 3:
            domains.add(parts[2])
    domains = sorted(domains)
    print(f"  Unique domains: {len(domains)}")

    # Test 2: Single domain
    print("\n[2] Single domain: hockey")
    print("-" * 40)
    s = FastAPIMCPServer(fastapi_base_url=url, domains=["hockey"])
    await s.initialize()
    print(f"  Tools: {len(s.tools_cache)}")

    # Test 3: Multiple domains
    print("\n[3] Multiple domains: hockey, movie, financial")
    print("-" * 40)
    s = FastAPIMCPServer(fastapi_base_url=url, domains=["hockey", "movie", "financial"])
    await s.initialize()
    print(f"  Tools: {len(s.tools_cache)}")

    # Test 4: Every domain individually
    print("\n[4] Tool count per domain")
    print("-" * 40)
    total = 0
    for domain in domains:
        s = FastAPIMCPServer(fastapi_base_url=url, domains=[domain])
        await s.initialize()
        count = len(s.tools_cache)
        total += count
        print(f"  {domain:35s} {count:4d} tools")

    print(f"\n  Sum of all domains: {total}")
    print(f"  Total (unfiltered): {len(server_all.tools_cache)}")


if __name__ == "__main__":
    asyncio.run(test_filtering())
