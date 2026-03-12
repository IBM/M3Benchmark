"""
Index all domain JSON files into separate ChromaDB collections
and generate sample query files per domain.

Only domains listed as keys in domain_description_dict.json are indexed.
Domains present in the source directory but absent from that file are skipped.

Usage:
    python index_all_domains.py /path/to/docs_by_domains_20k
    python index_all_domains.py /path/to/docs_by_domains_20k --max-domains 5
"""

import argparse
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

from chromadb_retriever import ChromaDBRetriever

PERSIST_DIR = "./chroma_data_52_domains"
QUERIES_DIR = "./queries_52_domains"
DOMAIN_DESCRIPTION_FILE = Path(__file__).parent / "domain_description_dict.json"


def extract_titles(chunks: list[dict]) -> list[str]:
    """Extract unique document titles from chunk texts.

    Each chunk's text starts with a title before the first newline.
    """
    titles = []
    seen = set()
    for chunk in chunks:
        title = chunk["text"].split("\n", 1)[0].strip()
        if title and title not in seen:
            seen.add(title)
            titles.append(title)
    return titles


def extract_snippets(chunks: list[dict], max_snippets: int = 30) -> list[str]:
    """Extract short content snippets (first sentence after the title) from chunks."""
    snippets = []
    seen = set()
    for chunk in chunks:
        parts = chunk["text"].split("\n", 1)
        if len(parts) < 2:
            continue
        body = parts[1].strip()
        # Take the first sentence
        for end in [".", "!", "?"]:
            idx = body.find(end)
            if idx != -1:
                sentence = body[: idx + 1].strip()
                if len(sentence) > 20 and sentence not in seen:
                    seen.add(sentence)
                    snippets.append(sentence)
                break
        if len(snippets) >= max_snippets:
            break
    return snippets


def generate_queries(titles: list[str], snippets: list[str], n: int = 50) -> list[dict]:
    """Generate diverse queries from document titles and content snippets."""
    # Templates that take a title
    title_templates = [
        "What is {}?",
        "Tell me about {}.",
        "How does {} work?",
        "What are the key facts about {}?",
        "Explain {}.",
        "Describe {}.",
        "Who or what is {}?",
        "Give me a summary of {}.",
        "What do you know about {}?",
        "What is the history of {}?",
        "Why is {} important?",
        "What is the significance of {}?",
        "What role does {} play?",
        "What happened with {}?",
        "Can you explain {} in detail?",
    ]

    # Templates that take a snippet (content-based queries)
    snippet_templates = [
        "{}",  # use the snippet directly as a query
        "What does this refer to: {}",
        "Find information related to: {}",
    ]

    queries = []
    used_questions = set()

    # Phase 1: title-based queries (use up to 30 titles)
    sampled_titles = random.sample(titles, min(35, len(titles)))
    for title in sampled_titles:
        template = random.choice(title_templates)
        question = template.format(title)
        if question not in used_questions:
            used_questions.add(question)
            queries.append({"question": question, "expected_topic": title})
        if len(queries) >= 35:
            break

    # Phase 2: snippet-based queries (fill remaining slots)
    sampled_snippets = random.sample(snippets, min(20, len(snippets)))
    for snippet in sampled_snippets:
        # Truncate long snippets
        short = snippet[:120] + "..." if len(snippet) > 120 else snippet
        template = random.choice(snippet_templates)
        question = template.format(short)
        if question not in used_questions:
            used_questions.add(question)
            queries.append({"question": question, "expected_topic": ""})
        if len(queries) >= n:
            break

    # If we still need more, generate more title queries with remaining titles
    if len(queries) < n:
        remaining_titles = [t for t in titles if t not in {q["expected_topic"] for q in queries}]
        for title in remaining_titles:
            template = random.choice(title_templates)
            question = template.format(title)
            if question not in used_questions:
                used_questions.add(question)
                queries.append({"question": question, "expected_topic": title})
            if len(queries) >= n:
                break

    return queries[:n]


def prepare_chunks(raw_chunks: list[dict]) -> list[dict]:
    """Convert raw JSON chunks to the format expected by ChromaDBRetriever."""
    prepared = []
    for i, chunk in enumerate(raw_chunks):
        prepared.append({
            "id": f"{chunk['id']}_{i}",
            "text": chunk["text"],
            "metadata": {
                "doc_id": str(chunk["id"]),
                "from_clapnq": chunk.get("from_clapnq", False),
            },
        })
    return prepared


def index_domain(docs_path: Path, domain: str, max_chunks: int | None = None) -> int:
    """Index a single domain's chunks into its own ChromaDB collection."""
    with open(docs_path) as f:
        raw_chunks = json.load(f)

    if max_chunks and len(raw_chunks) > max_chunks:
        print(f"  Loaded {len(raw_chunks)} chunks, limiting to {max_chunks}")
        raw_chunks = raw_chunks[:max_chunks]
    else:
        print(f"  Loaded {len(raw_chunks)} chunks")

    # Index
    retriever = ChromaDBRetriever(collection_name=domain, persist_directory=PERSIST_DIR)
    prepared = prepare_chunks(raw_chunks)
    retriever.add_chunks(prepared)

    # Generate queries
    titles = extract_titles(raw_chunks)
    snippets = extract_snippets(raw_chunks)
    queries = generate_queries(titles, snippets)
    queries_path = Path(QUERIES_DIR) / f"{domain}_queries.json"
    with open(queries_path, "w") as f:
        json.dump(queries, f, indent=2)
    print(f"  Wrote {len(queries)} queries to {queries_path}")

    return len(raw_chunks)


def main():
    parser = argparse.ArgumentParser(description="Index domain JSON files into ChromaDB collections")
    parser.add_argument("docs_directory", type=Path, help="Directory containing *_docs.json files")
    parser.add_argument("--max-domains", type=int, default=None, help="Max number of domains to index (default: all)")
    parser.add_argument("--max-chunks", type=int, default=None, help="Max chunks to index per domain (default: all)")
    args = parser.parse_args()

    if not args.docs_directory.is_dir():
        parser.error(f"{args.docs_directory} is not a directory")

    # Load the allowed domain set from domain_description_dict.json
    with open(DOMAIN_DESCRIPTION_FILE) as f:
        domain_descriptions = json.load(f)
    allowed_domains = set(domain_descriptions.keys())
    print(f"Allowed domains (from domain_description_dict.json): {len(allowed_domains)}")

    os.makedirs(QUERIES_DIR, exist_ok=True)

    all_json_files = sorted(args.docs_directory.glob("*_docs.json"))
    print(f"Found {len(all_json_files)} domain files in source directory")

    skipped_domains = []
    json_files = []
    for jf in all_json_files:
        domain = jf.stem.replace("_docs", "")
        if domain in allowed_domains:
            json_files.append(jf)
        else:
            skipped_domains.append(domain)

    print(f"Skipping {len(skipped_domains)} domains not in domain_description_dict.json: {skipped_domains}")

    if args.max_domains:
        json_files = json_files[: args.max_domains]
        print(f"Limiting to first {args.max_domains} domains")

    print()

    started_at = datetime.now(timezone.utc)
    total_chunks = 0
    domain_stats = []
    for i, json_file in enumerate(json_files, 1):
        domain = json_file.stem.replace("_docs", "")
        print(f"[{i}/{len(json_files)}] Indexing domain: {domain}")
        count = index_domain(json_file, domain, max_chunks=args.max_chunks)
        total_chunks += count
        domain_stats.append({"domain": domain, "chunks_indexed": count})
        print()

    finished_at = datetime.now(timezone.utc)

    print("=" * 60)
    print(f"Done! Indexed {total_chunks} total chunks across {len(json_files)} domains.")
    print(f"Collections persisted to: {PERSIST_DIR}")
    print(f"Query files written to: {QUERIES_DIR}")

    # Write stats file
    stats = {
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "source_directory": str(args.docs_directory),
        "persist_directory": PERSIST_DIR,
        "queries_directory": QUERIES_DIR,
        "domain_description_file": str(DOMAIN_DESCRIPTION_FILE),
        "total_domain_files_found": len(all_json_files),
        "allowed_domains_count": len(allowed_domains),
        "domains_indexed_count": len(json_files),
        "domains_skipped_count": len(skipped_domains),
        "domains_skipped": sorted(skipped_domains),
        "total_chunks_indexed": total_chunks,
        "max_domains_arg": args.max_domains,
        "max_chunks_arg": args.max_chunks,
        "domains_indexed": domain_stats,
    }
    stats_path = Path(PERSIST_DIR) / "indexing_stats.json"
    os.makedirs(PERSIST_DIR, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats written to: {stats_path}")


if __name__ == "__main__":
    main()
