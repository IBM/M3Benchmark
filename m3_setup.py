#!/usr/bin/env python3
"""
Setup script for the M3 Enterprise Benchmark.

Downloads benchmark data from HuggingFace, pulls the unified Docker image,
and starts containers for all tasks.

Usage:
    # Install with init dependencies
    pip install -e ".[init]"

    # Run full setup (download data + pull image + start containers)
    python m3_setup.py

    # Individual steps
    python m3_setup.py --download-data
    python m3_setup.py --pull-image
    python m3_setup.py --start-containers
    python m3_setup.py --stop-containers
"""

import argparse
import getpass
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DOCKER_IMAGE = "docker.io/amurthi44g1wd/m3_environ:latest"
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"

# Each HuggingFace dataset repo maps to a subdirectory under data/
HF_DATASETS = {
    "anupamamurthi/db":         "db",
    # "anupamamurthi/tasks":      "tasks",
    "anupamamurthi/retriever-chroma-data": "chroma_data",
    "anupamamurthi/queries":    "queries",
}

# Container names must match apis/configs/mcp_connection_config.yaml
CONTAINERS = [
    "task_1_m3_environ",
    # "task_2_m3_environ",
    # "task_5_m3_environ",
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
def _ensure_hf_token() -> str:
    """Return a HuggingFace token, prompting the user if not already set."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        print("Using HuggingFace token from environment.")
        return token

    print("\nA HuggingFace token is required to download the benchmark data.")
    print("You can create one at: https://huggingface.co/settings/tokens")
    token = getpass.getpass("Enter your HuggingFace token: ").strip()
    if not token:
        print("Error: no token provided.")
        sys.exit(1)
    return token


def download_data() -> None:
    """Download benchmark data from HuggingFace dataset repos into data/."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub is not installed.")
        print("  pip install -e '.[init]'")
        sys.exit(1)

    token = _ensure_hf_token()

    print(f"\n=== Downloading data into {DATA_DIR} ===")
    print(f"Repos: {len(HF_DATASETS)}")

    for repo, subdir in HF_DATASETS.items():
        target = DATA_DIR / subdir
        print(f"\n--- {repo} -> data/{subdir}/ ---")
        snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            local_dir=str(target),
            token=token,
        )
        print(f"  [ok] {subdir}/")

    print("\nData download complete.")


def pull_image(image: str = DOCKER_IMAGE) -> None:
    """Pull the unified Docker image and tag it locally."""
    rt = _runtime()
    print(f"\n=== Pulling {image} ===")
    _run([rt, "pull", image], check=True)
    _run([rt, "tag", image, "m3_environ"], check=True)
    print("Image pulled and tagged as 'm3_environ'.")


def start_containers() -> None:
    """Start one container per task (names match mcp_connection_config.yaml)."""
    rt = _runtime()

    # Stop and remove all benchmark containers before starting fresh
    print("\n=== Cleaning up existing benchmark containers ===")
    for name in CONTAINERS:
        _run([rt, "kill", name], capture_output=True)
        _run([rt, "rm", "-f", name], capture_output=True)
    print("  Cleanup done.")

    print("\n=== Starting containers ===")

    db_dir = str(DATA_DIR / "db")
    configs_dir = str(PROJECT_ROOT / "apis" / "configs")
    chroma_dir = str(DATA_DIR / "chroma_data")
    queries_dir = str(DATA_DIR / "queries")

    for name in CONTAINERS:
        _run([
            rt, "run", "-d",
            "--name", name,
            "-v", f"{db_dir}:/app/db:ro",
            "-v", f"{configs_dir}:/app/apis/configs:ro",
            "-v", f"{chroma_dir}:/app/retrievers/chroma_data",
            "-v", f"{queries_dir}:/app/retrievers/queries:ro",
            "m3_environ",
        ], check=True)
        print(f"  Started {name}")

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
    print("\n=== Stopping containers ===")
    for name in CONTAINERS:
        _run([rt, "rm", "-f", name], capture_output=True)
        print(f"  Removed {name}")
    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Setup the M3 Enterprise Benchmark environment",
    )
    parser.add_argument(
        "--download-data", action="store_true",
        help="Download benchmark data from HuggingFace",
    )
    parser.add_argument(
        "--pull-image", action="store_true",
        help="Pull the unified Docker image",
    )
    parser.add_argument(
        "--start-containers", action="store_true",
        help="Start Docker containers for all tasks",
    )
    parser.add_argument(
        "--stop-containers", action="store_true",
        help="Stop and remove all benchmark containers",
    )
    parser.add_argument(
        "--docker-image", type=str, default=DOCKER_IMAGE,
        help=f"Docker image to pull (default: {DOCKER_IMAGE})",
    )

    args = parser.parse_args()

    # If no specific step requested, run all setup steps
    explicit = args.download_data or args.pull_image or args.start_containers or args.stop_containers

    if args.stop_containers:
        stop_containers()
        return

    if not explicit or args.download_data:
        download_data()

    if not explicit or args.pull_image:
        pull_image(args.docker_image)

    if not explicit or args.start_containers:
        start_containers()

    if not explicit:
        print("\n" + "=" * 60)
        print("Setup complete! You can now run the benchmark:")
        print("=" * 60)
        print()
        print("  # Single task, single domain")
        print("  python benchmark_runner.py --task_id 2 --run-agent --domain address")
        print()
        print("  # All three tasks for one domain")
        print("  python benchmark_runner.py --task_id 1 2 5 --run-agent --domain address")
        print()
        print("  # Parallel execution")
        print("  python benchmark_runner.py --task_id 1 2 5 --run-agent --domain address --parallel")
        print()
        print("  # Stop containers when done")
        print("  python m3_setup.py --stop-containers")
        print()


if __name__ == "__main__":
    main()
