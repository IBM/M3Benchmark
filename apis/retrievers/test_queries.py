"""
Test runner that loads query files and runs them against the correct domain retriever.

Includes:
- Topic match rate (does the expected topic appear in results?)
- Recall@K (test at K=1, 3, 5, 10)
- Distance distribution analysis
- Cross-domain leakage test
- Negative query test

Usage:
    # Test all domains
    python test_queries.py

    # Test specific domains
    python test_queries.py address hockey

    # Test with more results per query
    python test_queries.py --n-results 5 address

    # Run cross-domain leakage test
    python test_queries.py --cross-domain address hockey

    # Run Recall@K analysis
    python test_queries.py --recall-at-k address
"""

import argparse
import json
import sys
import time
from pathlib import Path

import chromadb
from chromadb_retriever import ChromaDBRetriever, GraniteEmbeddingFunction

PERSIST_DIR = "./chroma_data"
QUERIES_DIR = "./queries"


def get_retriever(domain: str, embedding_fn: GraniteEmbeddingFunction) -> ChromaDBRetriever:
    """Create a retriever for a specific domain."""
    retriever = ChromaDBRetriever.__new__(ChromaDBRetriever)
    retriever.embedding_fn = embedding_fn
    retriever.client = chromadb.PersistentClient(path=PERSIST_DIR)
    retriever.collection = retriever.client.get_or_create_collection(
        name=domain,
        embedding_function=embedding_fn,
    )
    return retriever


def load_queries(domain: str) -> list[dict]:
    path = Path(QUERIES_DIR) / f"{domain}_queries.json"
    if not path.exists():
        print(f"  WARNING: No query file found at {path}")
        return []
    with open(path) as f:
        return json.load(f)


# --------------- Test: Basic query + topic match ---------------

def test_domain(domain: str, embedding_fn: GraniteEmbeddingFunction, n_results: int = 3) -> dict:
    """Run all queries for a domain and return stats."""
    print(f"\n{'=' * 60}")
    print(f"Domain: {domain}")
    print(f"{'=' * 60}")

    retriever = get_retriever(domain, embedding_fn)
    chunk_count = retriever.collection.count()
    print(f"Collection '{domain}' has {chunk_count} chunks")

    if chunk_count == 0:
        print("  SKIP: Collection is empty (not indexed yet)")
        return {"domain": domain, "total": 0, "success": 0, "errors": 0, "topic_matches": 0}

    queries = load_queries(domain)
    if not queries:
        return {"domain": domain, "total": 0, "success": 0, "errors": 0, "topic_matches": 0}

    print(f"Running {len(queries)} queries...\n")

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
            results = retriever.query(question, n_results=n_results)
            elapsed = time.time() - start
            total_time += elapsed

            docs = results["documents"][0] if results["documents"] else []
            distances = results["distances"][0] if results["distances"] else []

            if docs:
                success += 1
                all_distances.extend(distances)
                top_doc = docs[0][:100] + "..." if len(docs[0]) > 100 else docs[0]
                top_dist = distances[0]

                topic_found = ""
                if expected_topic:
                    for doc in docs:
                        if expected_topic.lower() in doc.lower():
                            topic_found = " [TOPIC MATCH]"
                            topic_matches += 1
                            break

                print(f"  Q{i:02d} ({elapsed:.3f}s): {question[:80]}")
                print(f"       Top result (dist={top_dist:.4f}): {top_doc}{topic_found}")
            else:
                errors += 1
                print(f"  Q{i:02d} ({elapsed:.3f}s): {question[:80]}")
                print(f"       NO RESULTS")

        except Exception as e:
            errors += 1
            print(f"  Q{i:02d}: {question[:80]}")
            print(f"       ERROR: {e}")

    avg_time = total_time / len(queries) if queries else 0
    queries_with_topic = sum(1 for q in queries if q.get("expected_topic"))
    topic_rate = (topic_matches / queries_with_topic * 100) if queries_with_topic else 0

    # Distance stats
    dist_min = min(all_distances) if all_distances else 0
    dist_max = max(all_distances) if all_distances else 0
    dist_avg = sum(all_distances) / len(all_distances) if all_distances else 0

    print(f"\n--- {domain} summary ---")
    print(f"  Queries: {len(queries)} | Success: {success} | Errors: {errors}")
    print(f"  Topic match: {topic_matches}/{queries_with_topic} ({topic_rate:.1f}%)")
    print(f"  Distance: min={dist_min:.4f}  avg={dist_avg:.4f}  max={dist_max:.4f}")
    print(f"  Avg query time: {avg_time:.3f}s")

    return {
        "domain": domain,
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


# --------------- Test: Recall@K ---------------

def test_recall_at_k(domain: str, embedding_fn: GraniteEmbeddingFunction) -> None:
    """Test recall at different K values (1, 3, 5, 10)."""
    print(f"\n{'=' * 60}")
    print(f"Recall@K Analysis: {domain}")
    print(f"{'=' * 60}")

    retriever = get_retriever(domain, embedding_fn)
    queries = load_queries(domain)

    # Only use queries with expected topics
    queries = [q for q in queries if q.get("expected_topic")]
    if not queries:
        print("  No queries with expected_topic found.")
        return

    print(f"Testing {len(queries)} queries with expected topics\n")

    k_values = [1, 3, 5, 10]
    recall = {k: 0 for k in k_values}

    for q in queries:
        question = q["question"]
        expected_topic = q["expected_topic"].lower()

        results = retriever.query(question, n_results=max(k_values))
        docs = results["documents"][0] if results["documents"] else []

        for k in k_values:
            top_k_docs = docs[:k]
            if any(expected_topic in doc.lower() for doc in top_k_docs):
                recall[k] += 1

    print(f"  {'K':>5}  {'Recall':>10}  {'Rate':>8}")
    print(f"  {'-' * 27}")
    for k in k_values:
        rate = recall[k] / len(queries) * 100
        bar = "#" * int(rate / 2)
        print(f"  {k:>5}  {recall[k]:>5}/{len(queries):<4}  {rate:>6.1f}%  {bar}")


# --------------- Test: Cross-domain leakage ---------------

def test_cross_domain(source_domain: str, target_domain: str, embedding_fn: GraniteEmbeddingFunction) -> None:
    """Query one domain's collection with another domain's queries. Distances should be high."""
    print(f"\n{'=' * 60}")
    print(f"Cross-Domain Test: '{source_domain}' queries -> '{target_domain}' collection")
    print(f"{'=' * 60}")

    retriever = get_retriever(target_domain, embedding_fn)
    queries = load_queries(source_domain)

    if not queries:
        print("  No queries found.")
        return

    # Use a subset
    test_queries = queries[:10]
    distances = []

    for q in test_queries:
        results = retriever.query(q["question"], n_results=1)
        if results["distances"] and results["distances"][0]:
            distances.append(results["distances"][0][0])

    if distances:
        avg_dist = sum(distances) / len(distances)
        min_dist = min(distances)
        max_dist = max(distances)
        print(f"  Tested {len(test_queries)} queries")
        print(f"  Distances: min={min_dist:.4f}  avg={avg_dist:.4f}  max={max_dist:.4f}")
        print(f"  (Higher distances = better isolation between domains)")

    # Now compare with in-domain distances
    own_retriever = get_retriever(source_domain, embedding_fn)
    own_distances = []
    for q in test_queries:
        results = own_retriever.query(q["question"], n_results=1)
        if results["distances"] and results["distances"][0]:
            own_distances.append(results["distances"][0][0])

    if own_distances and distances:
        own_avg = sum(own_distances) / len(own_distances)
        print(f"\n  Comparison:")
        print(f"    In-domain  ({source_domain:>20} -> {source_domain:<20}): avg dist = {own_avg:.4f}")
        print(f"    Cross-domain ({source_domain:>18} -> {target_domain:<20}): avg dist = {avg_dist:.4f}")
        ratio = avg_dist / own_avg if own_avg > 0 else 0
        print(f"    Ratio: {ratio:.2f}x (higher = better isolation)")


# --------------- Test: Negative queries ---------------

def test_negative_queries(domain: str, embedding_fn: GraniteEmbeddingFunction) -> None:
    """Query with completely unrelated content. Distances should be high."""
    print(f"\n{'=' * 60}")
    print(f"Negative Query Test: {domain}")
    print(f"{'=' * 60}")

    retriever = get_retriever(domain, embedding_fn)

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
        results = retriever.query(q, n_results=1)
        if results["distances"] and results["distances"][0]:
            dist = results["distances"][0][0]
            distances.append(dist)
            top_doc = results["documents"][0][0][:80] + "..." if results["documents"][0] else "N/A"
            print(f"    dist={dist:.4f}  Q: {q[:60]}")
            print(f"             R: {top_doc}")

    if distances:
        avg_dist = sum(distances) / len(distances)
        print(f"\n  Avg distance for unrelated queries: {avg_dist:.4f}")
        print(f"  (This is your 'no relevant results' baseline threshold)")


# --------------- Main ---------------

def main():
    parser = argparse.ArgumentParser(description="Test retriever queries per domain")
    parser.add_argument("domains", nargs="*", help="Domains to test (default: all)")
    parser.add_argument("--n-results", type=int, default=3, help="Number of results per query")
    parser.add_argument("--recall-at-k", action="store_true", help="Run Recall@K analysis")
    parser.add_argument("--cross-domain", nargs=2, metavar=("SOURCE", "TARGET"),
                        help="Test cross-domain leakage (SOURCE queries -> TARGET collection)")
    parser.add_argument("--negative", action="store_true", help="Run negative query test")
    args = parser.parse_args()

    # Load embedding model once
    print("Loading Granite embedding model...")
    embedding_fn = GraniteEmbeddingFunction()

    # Cross-domain test mode
    if args.cross_domain:
        test_cross_domain(args.cross_domain[0], args.cross_domain[1], embedding_fn)
        return

    # Discover domains
    if args.domains:
        domains = args.domains
    else:
        query_files = sorted(Path(QUERIES_DIR).glob("*_queries.json"))
        if not query_files:
            print(f"No query files found in {QUERIES_DIR}/")
            print("Run index_all_domains.py first to generate them.")
            sys.exit(1)
        domains = [f.stem.replace("_queries", "") for f in query_files]

    print(f"Testing {len(domains)} domain(s): {', '.join(domains)}")

    # Run tests per domain
    all_stats = []
    for domain in domains:
        stats = test_domain(domain, embedding_fn, n_results=args.n_results)
        all_stats.append(stats)

        if args.recall_at_k:
            test_recall_at_k(domain, embedding_fn)

        if args.negative:
            test_negative_queries(domain, embedding_fn)

    # Overall summary
    print(f"\n{'=' * 60}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 60}")

    header = f"{'Domain':<25} {'Queries':>7} {'OK':>5} {'Err':>5} {'TopicMatch':>11} {'AvgDist':>8} {'Time':>7}"
    print(header)
    print("-" * len(header))
    for s in all_stats:
        topic = f"{s['topic_matches']}/{s['topic_queries']}" if s["topic_queries"] else "N/A"
        avg_t = f"{s['avg_time_s']:.3f}s" if s["total"] > 0 else "N/A"
        dist = f"{s['dist_avg']:.4f}" if s["total"] > 0 else "N/A"
        print(f"{s['domain']:<25} {s['total']:>7} {s['success']:>5} {s['errors']:>5} {topic:>11} {dist:>8} {avg_t:>7}")

    print("-" * len(header))
    total_q = sum(s["total"] for s in all_stats)
    total_s = sum(s["success"] for s in all_stats)
    total_e = sum(s["errors"] for s in all_stats)
    total_tm = sum(s["topic_matches"] for s in all_stats)
    total_tq = sum(s["topic_queries"] for s in all_stats)
    overall_rate = f"{total_tm}/{total_tq}" if total_tq else "N/A"
    print(f"{'TOTAL':<25} {total_q:>7} {total_s:>5} {total_e:>5} {overall_rate:>11}")

    if total_tq:
        print(f"\nOverall topic match rate: {total_tm / total_tq * 100:.1f}%")


if __name__ == "__main__":
    main()
