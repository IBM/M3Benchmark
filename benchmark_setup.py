#!/usr/bin/env python3
"""
Setup script for the Enterprise Benchmark.

Downloads benchmark data from HuggingFace and manages benchmark containers
via docker compose.

Usage:
    # Install with init dependencies
    pip install -e ".[init]"

    # Run full setup (download data + start containers)
    python benchmark_setup.py

    # Individual steps
    python benchmark_setup.py --download-data
    python benchmark_setup.py --start-containers
    python benchmark_setup.py --stop-containers
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT

# Each HuggingFace dataset repo maps to a subdirectory under data/
HF_DATASETS = {
    "ibm-research/VAKRA": "data"
}

# Container names must match benchmark/mcp_connection_config.yaml
CONTAINERS = [
    "capability_1_bi_apis",
    "capability_2_dashboard_apis",
    "capability_3_multihop_reasoning",   # BPO + REST (no retriever)
    "capability_4_multiturn",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _runtime() -> str:
    """Return 'docker' or 'podman', whichever is available."""
    for rt in ["docker", "podman"]:
        if shutil.which(rt):
            return rt
    print("Error: neither docker nor podman found on PATH.")
    sys.exit(1)


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command, printing it first."""
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, **kwargs)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def _load_metadata(path: Path) -> dict:
    """Load locally stored file metadata (filename -> blob sha)."""
    if path.exists():
        return json.loads(path.read_text())
    return {}


def download_data() -> None:
    """Download only changed/added/deleted files from HuggingFace dataset repos."""
    try:
        from huggingface_hub import HfApi, RepoFile, hf_hub_download
    except ImportError:
        print("Error: huggingface_hub is not installed.")
        print("  pip install -e '.[init]'")
        sys.exit(1)

    api = HfApi()

    print(f"\n=== Syncing data into {DATA_DIR} ===")
    print(f"Repos: {len(HF_DATASETS)}")

    for repo, subdir in HF_DATASETS.items():
        target = DATA_DIR / subdir
        target.mkdir(parents=True, exist_ok=True)
        metadata_path = target / ".hf_metadata.json"

        print(f"\n--- {repo} -> data/{subdir}/ ---")

        # Get remote file tree (filename -> blob sha)
        remote_files = {
            item.path: item.blob_id
            for item in api.list_repo_tree(repo_id=repo, repo_type="dataset", recursive=True)
            if isinstance(item, RepoFile)
        }

        local_metadata = _load_metadata(metadata_path)

        # Skip files inside a "train" directory (HuggingFace train split)
        def _in_train_dir(path: str) -> bool:
            parts = Path(path).parts
            return len(parts) > 0 and parts[0] == "train"

        to_download = [
            p for p, sha in remote_files.items()
            if local_metadata.get(p) != sha and not _in_train_dir(p)
        ]
        to_delete = [p for p in local_metadata if p not in remote_files and not _in_train_dir(p)]

        if not to_download and not to_delete:
            print(f"  [up to date] {subdir}/")
            continue

        for path in to_delete:
            local_path = target / path
            if local_path.exists():
                local_path.unlink()
            print(f"  [deleted] {path}")

        print(f"  Downloading {len(to_download)} file(s)...")
        for path in to_download:
            hf_hub_download(
                repo_id=repo,
                filename=path,
                repo_type="dataset",
                local_dir=str(target),

            )
            print(f"  [ok] {path}")

        metadata_path.write_text(json.dumps(remote_files, indent=2))

    print("\nData sync complete.")


def start_containers() -> None:
    """Start all benchmark containers via docker compose."""
    rt = _runtime()
    print("\n=== Starting containers ===")

    db_dir = DATA_DIR / "databases"
    if not db_dir.exists() or not any(db_dir.iterdir()):
        raise SystemExit(
            f"\nERROR: Database directory '{db_dir}' is missing or empty.\n"
            "Run 'make download' (or 'python benchmark_setup.py --download-data') first.\n"
        )

    _run([rt, "compose", "up", "-d"], check=True)

    # Wait for the internal FastAPI servers to come up
    print("\nWaiting for services to initialize (up to 120s) ...")
    deadline = time.time() + 120
    ready = set()
    while time.time() < deadline and len(ready) < len(CONTAINERS):
        for name in CONTAINERS:
            if name in ready:
                continue
            result = _run(
                [rt, "exec", name, "curl", "-sf", "http://localhost:8000/openapi.json"],
                capture_output=True,
            )
            if result.returncode == 0:
                ready.add(name)
                print(f"  [ready] {name}")
        if len(ready) < len(CONTAINERS):
            time.sleep(5)

    not_ready = set(CONTAINERS) - ready
    if not_ready:
        print(f"\nWarning: these containers did not become ready: {not_ready}")
        print("Check logs with:  docker logs <container_name>")
    else:
        print("\nAll containers are ready.")


def stop_containers() -> None:
    """Stop and remove all benchmark containers."""
    rt = _runtime()
    print("\n=== Stopping and removing benchmark containers ===")
    _run([rt, "compose", "down", "--remove-orphans"], capture_output=True)
    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Setup the Enterprise Benchmark environment",
    )
    parser.add_argument(
        "--download-data", action="store_true",
        help="Download benchmark data from HuggingFace",
    )
    parser.add_argument(
        "--start-containers", action="store_true",
        help="Start Docker containers for all tasks via docker compose",
    )
    parser.add_argument(
        "--stop-containers", action="store_true",
        help="Stop and remove all benchmark containers",
    )

    args = parser.parse_args()

    # If no specific step requested, run all setup steps
    explicit = args.download_data or args.start_containers or args.stop_containers

    if args.stop_containers:
        stop_containers()
        return

    if not explicit or args.download_data:
        download_data()

    if not explicit or args.start_containers:
        start_containers()

    if not explicit:
        print("\n" + "=" * 60)
        print("Setup complete! You can now run the benchmark:")
        print("=" * 60)
        print()
        print("  # Single task, single domain")
        print("  python benchmark_runner.py --capability_id 2 --run-agent --domain address")
        print()
        print("  # All three tasks for one domain")
        print("  python benchmark_runner.py --capability_id 1 2 4 --run-agent --domain address")
        print()
        print("  # Parallel execution")
        print("  python benchmark_runner.py --capability_id 1 2 4 --run-agent --domain address --parallel")
        print()
        print("  # Stop containers when done")
        print("  python benchmark_setup.py --stop-containers")
        print()


if __name__ == "__main__":
    main()
