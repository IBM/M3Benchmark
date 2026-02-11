"""
Unified test runner for ChromaDB retriever domains.

Three modes:
  1. fastapi (default) - Direct HTTP to FastAPI server at localhost:8001
  2. mcp               - In-process MCP server wrapping FastAPI
  3. docker            - MCP over stdio to Docker container

Usage:
    # FastAPI mode (default) - requires: python run.py
    python test_queries.py address --max-queries 5
    python test_queries.py address hockey

    # MCP mode - requires: python run.py
    python test_queries.py --mode mcp address --max-queries 5

    # Docker mode - requires: docker compose up
    python test_queries.py --mode docker address --max-queries 5

    # Extra tests (work with any mode)
    python test_queries.py --recall-at-k address
    python test_queries.py --negative address
    python test_queries.py --cross-domain address hockey

    # Test all domains
    python test_queries.py
    python test_queries.py --mode docker
"""

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx

QUERIES_DIR = "./queries"
DEFAULT_CONTAINER_NAME = "retriever-mcp-server"

EMPTY_STATS = {
    "tools": 0, "total": 0, "success": 0, "errors": 0,
    "topic_matches": 0, "topic_queries": 0, "topic_rate": 0,
    "dist_min": 0, "dist_avg": 0, "dist_max": 0, "avg_time_s": 0,
}


def load_queries(domain: str) -> list[dict]:
    path = Path(QUERIES_DIR) / f"{domain}_queries.json"
    if not path.exists():
        print(f"  WARNING: No query file found at {path}")
        return []
    with open(path) as f:
        return json.load(f)


def discover_domains() -> list[str]:
    query_files = sorted(Path(QUERIES_DIR).glob("*_queries.json"))
    if not query_files:
        print(f"No query files found in {QUERIES_DIR}/")
        print("Run index_all_domains.py first to generate them.")
        sys.exit(1)
    return [f.stem.replace("_queries", "") for f in query_files]


# --------------- Container helpers ---------------

def detect_container_runtime() -> str:
    """Detect available container runtime (podman or docker)."""
    if shutil.which("podman"):
        return "podman"
    if shutil.which("docker"):
        return "docker"
    raise RuntimeError("Neither podman nor docker found in PATH.")


def stop_mcp_server(container_runtime: str, container_name: str):
    """Stop any running mcp_server.py processes inside the container."""
    try:
        subprocess.run(
            [container_runtime, "exec", container_name, "pkill", "-f", "python mcp_server.py"],
            capture_output=True, timeout=5,
        )
    except Exception:
        pass


# --------------- Query function factories ---------------
# Each returns an async callable: (question, n_results) -> list[dict]
# where each dict has keys: id, text, distance

def make_fastapi_query_fn(domain: str, base_url: str):
    """Create a query function that calls FastAPI directly via HTTP."""
    async def query_fn(question: str, n_results: int = 3) -> list[dict]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{base_url}/{domain}/query",
                json={"question": question, "n_results": n_results},
            )
            resp.raise_for_status()
            return resp.json().get("results", [])
    return query_fn


async def make_mcp_query_fn(domain: str, base_url: str):
    """Create a query function using in-process MCP server.

    Returns (query_fn, num_tools) or (None, 0) on failure.
    """
    from mcp_server import FastAPIMCPServer

    mcp = FastAPIMCPServer(
        fastapi_base_url=base_url,
        server_name=f"retriever-mcp-{domain}",
        domains=[domain],
    )
    try:
        await mcp.initialize()
    except Exception as e:
        print(f"  ERROR: Could not connect to FastAPI server at {base_url}")
        print(f"  {e}")
        print(f"  Make sure the retriever server is running: python run.py")
        return None, 0

    tools = await mcp.list_tools()
    if not tools:
        print(f"  ERROR: No tools found for domain '{domain}'")
        return None, 0

    tool_name = tools[0].name
    print(f"  MCP tool: {tool_name}")

    async def query_fn(question: str, n_results: int = 3) -> list[dict]:
        results = await mcp.call_tool(
            tool_name,
            {"body": {"question": question, "n_results": n_results}},
        )
        response_text = results[0].text if results else ""
        data = json.loads(response_text)
        return data.get("results", [])

    return query_fn, len(tools)


# --------------- Shared test functions ---------------

async def run_basic_queries(
    query_fn, queries: list[dict],
    n_results: int = 3, max_queries: int | None = None,
) -> dict:
    """Run queries and return stats dict."""
    if max_queries:
        queries = queries[:max_queries]

    success = 0
    errors = 0
    topic_matches = 0
    total_time = 0.0
    all_distances = []

    for i, q in enumerate(queries, 1):
        question = q["question"]
        expected_topic = q.get("expected_topic", "")

        try:
            start = time.time()
            results = await query_fn(question, n_results)
            elapsed = time.time() - start
            total_time += elapsed

            if not results:
                errors += 1
                print(f"  Q{i:02d} ({elapsed:.3f}s): {question[:80]}")
                print(f"       NO RESULTS")
                continue

            success += 1
            top = results[0]
            top_text = top["text"][:80] + "..." if len(top["text"]) > 80 else top["text"]
            top_dist = top["distance"]
            all_distances.extend(r["distance"] for r in results)

            topic_found = ""
            if expected_topic:
                for r in results:
                    if expected_topic.lower() in r["text"].lower():
                        topic_found = " [TOPIC MATCH]"
                        topic_matches += 1
                        break

            print(f"  Q{i:02d} ({elapsed:.3f}s): {question[:80]}")
            print(f"       Top (dist={top_dist:.4f}): {top_text}{topic_found}")
        except Exception as e:
            errors += 1
            print(f"  Q{i:02d}: {question[:80]}")
            print(f"       ERROR: {e}")

    queries_with_topic = sum(1 for q in queries if q.get("expected_topic"))
    topic_rate = (topic_matches / queries_with_topic * 100) if queries_with_topic else 0
    avg_time = total_time / len(queries) if queries else 0

    dist_min = min(all_distances) if all_distances else 0
    dist_max = max(all_distances) if all_distances else 0
    dist_avg = sum(all_distances) / len(all_distances) if all_distances else 0

    return {
        "total": len(queries),
        "success": success,
        "errors": errors,
        "topic_matches": topic_matches,
        "topic_queries": queries_with_topic,
        "topic_rate": round(topic_rate, 1),
        "dist_min": round(dist_min, 4),
        "dist_avg": round(dist_avg, 4),
        "dist_max": round(dist_max, 4),
        "avg_time_s": round(avg_time, 3),
    }


async def test_recall_at_k(query_fn, queries: list[dict]):
    """Test recall at different K values (1, 3, 5, 10)."""
    queries = [q for q in queries if q.get("expected_topic")]
    if not queries:
        print("  No queries with expected_topic found.")
        return

    print(f"\nRecall@K Analysis ({len(queries)} queries with expected topics)\n")

    k_values = [1, 3, 5, 10]
    recall = {k: 0 for k in k_values}

    for q in queries:
        question = q["question"]
        expected_topic = q["expected_topic"].lower()

        results = await query_fn(question, max(k_values))
        texts = [r["text"] for r in results]

        for k in k_values:
            top_k = texts[:k]
            if any(expected_topic in t.lower() for t in top_k):
                recall[k] += 1

    print(f"  {'K':>5}  {'Recall':>10}  {'Rate':>8}")
    print(f"  {'-' * 27}")
    for k in k_values:
        rate = recall[k] / len(queries) * 100
        bar = "#" * int(rate / 2)
        print(f"  {k:>5}  {recall[k]:>5}/{len(queries):<4}  {rate:>6.1f}%  {bar}")


async def test_cross_domain(
    source_domain: str, target_domain: str,
    query_fn_source, query_fn_target,
):
    """Query target domain with source domain's queries. Distances should be high."""
    print(f"\n{'=' * 60}")
    print(f"Cross-Domain Test: '{source_domain}' queries -> '{target_domain}' collection")
    print(f"{'=' * 60}")

    queries = load_queries(source_domain)
    if not queries:
        print("  No queries found.")
        return

    test_queries = queries[:10]
    cross_distances = []

    for q in test_queries:
        results = await query_fn_target(q["question"], 1)
        if results:
            cross_distances.append(results[0]["distance"])

    if cross_distances:
        avg_dist = sum(cross_distances) / len(cross_distances)
        min_dist = min(cross_distances)
        max_dist = max(cross_distances)
        print(f"  Tested {len(test_queries)} queries")
        print(f"  Distances: min={min_dist:.4f}  avg={avg_dist:.4f}  max={max_dist:.4f}")
        print(f"  (Higher distances = better isolation between domains)")

    # Compare with in-domain distances
    own_distances = []
    for q in test_queries:
        results = await query_fn_source(q["question"], 1)
        if results:
            own_distances.append(results[0]["distance"])

    if own_distances and cross_distances:
        own_avg = sum(own_distances) / len(own_distances)
        print(f"\n  Comparison:")
        print(f"    In-domain  ({source_domain:>20} -> {source_domain:<20}): avg dist = {own_avg:.4f}")
        print(f"    Cross-domain ({source_domain:>18} -> {target_domain:<20}): avg dist = {avg_dist:.4f}")
        ratio = avg_dist / own_avg if own_avg > 0 else 0
        print(f"    Ratio: {ratio:.2f}x (higher = better isolation)")


async def test_negative_queries(query_fn, domain: str):
    """Query with completely unrelated content. Distances should be high."""
    print(f"\n{'=' * 60}")
    print(f"Negative Query Test: {domain}")
    print(f"{'=' * 60}")

    garbage_queries = [
        "recipe for chocolate lava cake with raspberry sauce",
        "how to knit a sweater for beginners",
        "best yoga poses for back pain relief",
        "quantum entanglement in photosynthesis",
        "plumbing repair for leaky kitchen faucet",
    ]

    print(f"  Running {len(garbage_queries)} unrelated queries:\n")
    distances = []

    for q in garbage_queries:
        results = await query_fn(q, 1)
        if results:
            dist = results[0]["distance"]
            distances.append(dist)
            top_text = results[0]["text"][:80] + "..."
            print(f"    dist={dist:.4f}  Q: {q[:60]}")
            print(f"             R: {top_text}")

    if distances:
        avg_dist = sum(distances) / len(distances)
        print(f"\n  Avg distance for unrelated queries: {avg_dist:.4f}")
        print(f"  (This is your 'no relevant results' baseline threshold)")


# --------------- Per-domain test orchestration ---------------

def print_domain_summary(domain: str, stats: dict):
    """Print per-domain summary."""
    print(f"\n--- {domain} summary ---")
    print(f"  Mode: {stats.get('mode', 'unknown')}")
    if stats.get("tools"):
        print(f"  Tools/endpoints: {stats['tools']}")
    print(f"  Queries: {stats['total']} | Success: {stats['success']} | Errors: {stats['errors']}")
    print(f"  Topic match: {stats['topic_matches']}/{stats['topic_queries']} ({stats['topic_rate']:.1f}%)")
    if stats.get("dist_avg"):
        print(f"  Distance: min={stats['dist_min']:.4f}  avg={stats['dist_avg']:.4f}  max={stats['dist_max']:.4f}")
    print(f"  Avg query time: {stats['avg_time_s']:.3f}s")


async def test_domain(domain: str, args) -> dict:
    """Test a single domain using the specified mode."""
    mode = args.mode

    print(f"\n{'=' * 60}")
    print(f"Testing domain='{domain}' [mode: {mode}]")
    print(f"{'=' * 60}")

    queries = load_queries(domain)
    if not queries:
        return {"domain": domain, "mode": mode, **EMPTY_STATS}

    if mode == "fastapi":
        query_fn = make_fastapi_query_fn(domain, args.base_url)
        num_tools = 1
        # Verify connectivity
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{args.base_url}/health")
                resp.raise_for_status()
        except Exception as e:
            print(f"  ERROR: Could not connect to FastAPI server at {args.base_url}")
            print(f"  {e}")
            print(f"  Make sure the retriever server is running: python run.py")
            return {"domain": domain, "mode": mode, **EMPTY_STATS}
        print(f"  Endpoint: POST {args.base_url}/{domain}/query")

    elif mode == "mcp":
        query_fn, num_tools = await make_mcp_query_fn(domain, args.base_url)
        if query_fn is None:
            return {"domain": domain, "mode": mode, **EMPTY_STATS}

    elif mode == "docker":
        return await _test_domain_docker(domain, queries, args)

    # Run queries (fastapi and mcp modes)
    n_queries = min(len(queries), args.max_queries) if args.max_queries else len(queries)
    print(f"\n  Running {n_queries} queries (n_results={args.n_results})...\n")

    stats = await run_basic_queries(query_fn, queries, args.n_results, args.max_queries)

    if args.recall_at_k:
        await test_recall_at_k(query_fn, queries)
    if args.negative:
        await test_negative_queries(query_fn, domain)

    result = {"domain": domain, "mode": mode, "tools": num_tools, **stats}
    print_domain_summary(domain, result)
    return result


async def _test_domain_docker(domain: str, queries: list[dict], args) -> dict:
    """Docker mode: manages MCP session lifecycle and runs tests within it."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    container_runtime = args.container_runtime or detect_container_runtime()
    exec_args = [
        "exec", "-i",
        "-e", f"MCP_DOMAIN={domain}",
        args.container_name,
        "python", "mcp_server.py",
    ]
    print(f"  Starting MCP server: {container_runtime} {' '.join(exec_args)}")

    server_params = StdioServerParameters(
        command=container_runtime, args=exec_args, env=None,
    )

    result = {"domain": domain, "mode": "docker", **EMPTY_STATS}

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                response = await session.list_tools()
                num_tools = len(response.tools)
                print(f"  Found {num_tools} tool(s):")
                for tool in response.tools:
                    print(f"    - {tool.name}: {tool.description or ''}")

                if not response.tools:
                    print(f"  ERROR: No tools found for domain '{domain}'")
                    return result

                tool_name = response.tools[0].name

                async def query_fn(question: str, n_results: int = 3) -> list[dict]:
                    res = await session.call_tool(
                        tool_name,
                        {"body": {"question": question, "n_results": n_results}},
                    )
                    response_text = res.content[0].text if res.content else ""
                    data = json.loads(response_text)
                    return data.get("results", [])

                n_queries = min(len(queries), args.max_queries) if args.max_queries else len(queries)
                print(f"\n  Running {n_queries} queries (n_results={args.n_results})...\n")

                stats = await run_basic_queries(query_fn, queries, args.n_results, args.max_queries)

                if args.recall_at_k:
                    await test_recall_at_k(query_fn, queries)
                if args.negative:
                    await test_negative_queries(query_fn, domain)

                result = {"domain": domain, "mode": "docker", "tools": num_tools, **stats}
                print_domain_summary(domain, result)

        print("  Server stopped.")
    except ExceptionGroup as eg:
        print(f"  Warning: Cleanup error (ignored): {eg}")
    except Exception as e:
        if "TaskGroup" in str(type(e).__name__) or "TaskGroup" in str(e):
            print(f"  Warning: Cleanup error (ignored): {e}")
        else:
            stop_mcp_server(container_runtime, args.container_name)
            raise

    return result


# --------------- Main ---------------

async def run(args):
    # Handle cross-domain test separately
    if args.cross_domain:
        source, target = args.cross_domain
        if args.mode == "docker":
            print("Cross-domain test is not supported in docker mode.")
            print("Use --mode fastapi or --mode mcp instead.")
            sys.exit(1)
        if args.mode == "mcp":
            query_fn_source, _ = await make_mcp_query_fn(source, args.base_url)
            query_fn_target, _ = await make_mcp_query_fn(target, args.base_url)
        else:
            query_fn_source = make_fastapi_query_fn(source, args.base_url)
            query_fn_target = make_fastapi_query_fn(target, args.base_url)
        if query_fn_source and query_fn_target:
            await test_cross_domain(source, target, query_fn_source, query_fn_target)
        return

    # Discover domains
    domains = args.domains if args.domains else discover_domains()

    print(f"Mode: {args.mode}")
    print(f"Testing {len(domains)} domain(s): {', '.join(domains)}")

    if args.mode == "docker":
        container_runtime = args.container_runtime or detect_container_runtime()
        print(f"Container runtime: {container_runtime}")
        print(f"Container name: {args.container_name}")
    else:
        print(f"FastAPI server: {args.base_url}")

    all_stats = []
    for domain in domains:
        stats = await test_domain(domain, args)
        all_stats.append(stats)

    # Overall summary
    print(f"\n{'=' * 60}")
    print(f"OVERALL SUMMARY (mode: {args.mode})")
    print(f"{'=' * 60}")

    header = f"{'Domain':<25} {'Tools':>5} {'Queries':>7} {'OK':>5} {'Err':>5} {'TopicMatch':>11} {'AvgDist':>8} {'Time':>7}"
    print(header)
    print("-" * len(header))
    for s in all_stats:
        topic = f"{s['topic_matches']}/{s['topic_queries']}" if s.get("topic_queries") else "N/A"
        avg_t = f"{s['avg_time_s']:.3f}s" if s["total"] > 0 else "N/A"
        dist = f"{s['dist_avg']:.4f}" if s["total"] > 0 and s.get("dist_avg") else "N/A"
        tools = s.get("tools", "")
        print(f"{s['domain']:<25} {tools:>5} {s['total']:>7} {s['success']:>5} {s['errors']:>5} {topic:>11} {dist:>8} {avg_t:>7}")

    print("-" * len(header))
    total_q = sum(s["total"] for s in all_stats)
    total_s = sum(s["success"] for s in all_stats)
    total_e = sum(s["errors"] for s in all_stats)
    total_tm = sum(s["topic_matches"] for s in all_stats)
    total_tq = sum(s.get("topic_queries", 0) for s in all_stats)
    overall_topic = f"{total_tm}/{total_tq}" if total_tq else "N/A"
    print(f"{'TOTAL':<25} {'':>5} {total_q:>7} {total_s:>5} {total_e:>5} {overall_topic:>11}")

    if total_tq:
        print(f"\nOverall topic match rate: {total_tm / total_tq * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Test retriever queries per domain")
    parser.add_argument("domains", nargs="*", help="Domains to test (default: all)")
    parser.add_argument(
        "--mode", choices=["fastapi", "mcp", "docker"], default="fastapi",
        help="Test mode: 'fastapi' (direct HTTP), 'mcp' (in-process MCP server), "
             "'docker' (MCP via container) (default: fastapi)",
    )
    parser.add_argument("--max-queries", type=int, default=None, help="Max queries per domain")
    parser.add_argument("--n-results", type=int, default=3, help="Number of results per query")

    # Extra tests
    parser.add_argument("--recall-at-k", action="store_true", help="Run Recall@K analysis")
    parser.add_argument("--cross-domain", nargs=2, metavar=("SOURCE", "TARGET"),
                        help="Test cross-domain leakage (SOURCE queries -> TARGET collection)")
    parser.add_argument("--negative", action="store_true", help="Run negative query test")

    # Server options
    parser.add_argument(
        "--base-url", default="http://localhost:8001",
        help="FastAPI base URL (default: http://localhost:8001)",
    )

    # Docker mode options
    parser.add_argument(
        "--container-runtime", type=str, default=None,
        help="Container runtime: docker or podman (default: auto-detect)",
    )
    parser.add_argument(
        "--container-name", type=str, default=DEFAULT_CONTAINER_NAME,
        help=f"Container name for docker mode (default: {DEFAULT_CONTAINER_NAME})",
    )

    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
