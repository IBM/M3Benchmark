"""
Upload/download ChromaDB data and query files to/from Hugging Face Hub.

Pushes the pre-built ChromaDB collections so other machines can skip
the slow indexing step and start querying immediately.

Usage:
    # Upload (requires: huggingface-cli login)
    python hf_sync.py upload --repo your-org/retriever-chroma-data

    # Upload a specific variant
    python hf_sync.py upload --repo your-org/retriever-chroma-data --chroma-dir ./chroma_data_5k

    # Download
    python hf_sync.py download --repo your-org/retriever-chroma-data

    # Download to a custom location
    python hf_sync.py download --repo your-org/retriever-chroma-data --output-dir ./chroma-data

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login
"""

import argparse
import sys
from pathlib import Path


def upload(args):
    """Upload chroma_data/ and queries/ to a HuggingFace dataset repo."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Error: huggingface_hub not installed.")
        print("  pip install huggingface_hub")
        sys.exit(1)

    api = HfApi()
    repo_id = args.repo
    chroma_dir = Path(args.chroma_dir)
    queries_dir = Path(args.queries_dir)

    if not chroma_dir.exists():
        print(f"Error: {chroma_dir} does not exist.")
        print("Run index_all_domains.py first to create the ChromaDB collections.")
        sys.exit(1)

    # Create the repo if it doesn't exist
    print(f"Ensuring repo exists: {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    # Upload chroma_data
    print(f"\nUploading {chroma_dir} -> chroma_data/")
    print("This may take a while for large collections...")
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(chroma_dir),
        path_in_repo="chroma_data",
        repo_type="dataset",
        commit_message=f"Upload ChromaDB collections from {chroma_dir.name}",
    )
    print("  Done.")

    # Upload queries
    if queries_dir.exists():
        print(f"\nUploading {queries_dir} -> queries/")
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(queries_dir),
            path_in_repo="queries",
            repo_type="dataset",
            commit_message="Upload query files",
        )
        print("  Done.")
    else:
        print(f"\nSkipping queries (not found at {queries_dir})")

    print(f"\nUpload complete: https://huggingface.co/datasets/{repo_id}")


def download(args):
    """Download chroma_data/ and queries/ from a HuggingFace dataset repo."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub not installed.")
        print("  pip install huggingface_hub")
        sys.exit(1)

    repo_id = args.repo
    output_dir = Path(args.output_dir)

    print(f"Downloading {repo_id} -> {output_dir}/")
    print("This may take a while for large collections...")

    local_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(output_dir),
    )

    print(f"\nDownloaded to: {local_path}")

    chroma_path = output_dir / "chroma_data"
    queries_path = output_dir / "queries"

    if chroma_path.exists():
        print(f"  ChromaDB data: {chroma_path}")
    if queries_path.exists():
        print(f"  Query files:   {queries_path}")

    print(f"\nTo use with the server:")
    print(f"  cd apis/retrievers")
    if str(output_dir) != ".":
        print(f"  ln -sf {chroma_path.resolve()} ./chroma_data")
        print(f"  ln -sf {queries_path.resolve()} ./queries")
    print(f"  python run.py")


def main():
    parser = argparse.ArgumentParser(
        description="Sync ChromaDB data with Hugging Face Hub"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Upload
    up = subparsers.add_parser("upload", help="Upload to HuggingFace")
    up.add_argument("--repo", required=True, help="HF repo ID (e.g. your-org/retriever-chroma-data)")
    up.add_argument("--chroma-dir", default="./chroma_data", help="ChromaDB directory to upload (default: ./chroma_data)")
    up.add_argument("--queries-dir", default="./queries", help="Queries directory to upload (default: ./queries)")

    # Download
    down = subparsers.add_parser("download", help="Download from HuggingFace")
    down.add_argument("--repo", required=True, help="HF repo ID (e.g. your-org/retriever-chroma-data)")
    down.add_argument("--output-dir", default=".", help="Where to download (default: current directory)")

    args = parser.parse_args()

    if args.command == "upload":
        upload(args)
    elif args.command == "download":
        download(args)


if __name__ == "__main__":
    main()
