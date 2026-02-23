"""End-to-end tests for the M3 benchmark pipeline.

Tests the full pipeline:
  1. Download data from HuggingFace (incremental sync)
  2. Start Docker containers
  3. Run benchmark_runner.py for task 1 and task 2
  4. Validate the output JSON files

Requirements:
    HF_TOKEN       - HuggingFace token for data download
    OPENAI_API_KEY - Required because tests use --provider openai
    Docker         - Available on PATH

Usage:
    # Install setup dependencies first (needed for HuggingFace data download)
    pip install -e '.[init]'

    # Minimal — set env vars inline
    # Use 'python -m pytest' to ensure the active venv's Python is used
    HF_TOKEN=hf_... OPENAI_API_KEY=sk-... python -m pytest tests/e2e/test_benchmark_e2e.py -v -s

    # Using a .env file (copy template_env → .env and fill in values)
    cp template_env .env
    # edit .env: set HF_TOKEN and OPENAI_API_KEY
    export $(grep -v '^#' .env | xargs)
    python -m pytest tests/e2e/test_benchmark_e2e.py -v -s

    # Run only one task
    python -m pytest tests/e2e/test_benchmark_e2e.py::TestBenchmarkE2E::test_task1_address -v -s
    python -m pytest tests/e2e/test_benchmark_e2e.py::TestBenchmarkE2E::test_task2_address -v -s

Notes:
    - The module-scoped fixture downloads data and starts containers once,
      then tears them down after both tests finish.
    - Each test gets an isolated tmp_path for its output files.
    - All tests are skipped automatically when the required env vars are absent.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent


# ---------------------------------------------------------------------------
# Guard: skip the entire module when required env vars are absent
# ---------------------------------------------------------------------------

def _missing_env_vars() -> list[str]:
    missing = []
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        missing.append("HF_TOKEN")
    if not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    return missing


_MISSING = _missing_env_vars()
pytestmark = pytest.mark.skipif(
    bool(_MISSING),
    reason=f"Required environment variables not set: {', '.join(_MISSING)}",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def containers_ready():
    """Download benchmark data and start Docker containers.

    Runs once for the whole module; tears down containers on exit.
    """
    sys.path.insert(0, str(PROJECT_ROOT))
    from m3_setup import download_data, start_containers, stop_containers

    download_data()
    start_containers()
    yield
    # stop_containers()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_benchmark(task_id: int, output_dir: Path) -> subprocess.CompletedProcess:
    """Invoke benchmark_runner.py as a subprocess, streaming output to the terminal."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "benchmark_runner.py"),
        "--m3_task_id", str(task_id),
        "--domain", "address",
        "--provider", "openai",
        "--max-samples-per-domain", "2",
        "--output", str(output_dir),
    ]
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n", flush=True)
    return subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        timeout=300,  # 5 minutes — generous for 2 samples
    )


def _assert_output(output_dir: Path, domain: str = "address") -> list:
    """Assert the output file is valid and return the parsed records."""
    output_file = output_dir / f"{domain}.json"
    assert output_file.exists(), (
        f"Expected output file not found: {output_file}\n"
        f"Directory contents: {list(output_dir.iterdir())}"
    )

    with open(output_file) as fh:
        records = json.load(fh)

    assert isinstance(records, list), "Output must be a JSON array"
    assert 1 <= len(records) <= 2, (
        f"Expected 1-2 records (max-samples-per-domain=2), got {len(records)}"
    )

    for i, record in enumerate(records):
        ctx = f"records[{i}]"

        # Required top-level keys
        for key in ("uuid", "domain", "status", "duration_s", "ground_truth"):
            assert key in record, f"{ctx}: missing key '{key}'"

        # Domain matches what we requested
        assert record["domain"] == domain, (
            f"{ctx}: expected domain={domain!r}, got {record['domain']!r}"
        )

        # Status is one of the allowed values
        assert record["status"] in ("success", "error"), (
            f"{ctx}: unexpected status {record['status']!r}"
        )

        # duration_s is numeric
        assert isinstance(record["duration_s"], (int, float)), (
            f"{ctx}: 'duration_s' must be numeric, got {type(record['duration_s'])}"
        )

        # ground_truth is a non-empty list of turn records
        gt_list = record["ground_truth"]
        assert isinstance(gt_list, list) and len(gt_list) >= 1, (
            f"{ctx}: 'ground_truth' must be a non-empty list"
        )

        for j, gt in enumerate(gt_list):
            gt_ctx = f"{ctx}.ground_truth[{j}]"
            for key in ("query", "answer", "gold_sequence"):
                assert key in gt, f"{gt_ctx}: missing key '{key}'"
            assert isinstance(gt["gold_sequence"], list), (
                f"{gt_ctx}: 'gold_sequence' must be a list"
            )

    return records


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBenchmarkE2E:
    """Full end-to-end tests: containers → runner → output validation."""

    def test_task1_address(self, containers_ready, tmp_path):
        """Task 1: slot-filling agent on 2 address-domain samples."""
        output_dir = tmp_path / "task1"
        output_dir.mkdir()

        result = _run_benchmark(task_id=1, output_dir=output_dir)

        assert result.returncode == 0, (
            f"benchmark_runner.py exited with code {result.returncode} "
            f"(see output above)"
        )

        records = _assert_output(output_dir)

        successful = [r for r in records if r["status"] == "success"]
        assert len(successful) >= 1, (
            "Expected at least 1 successful record for task 1.\n"
            + json.dumps(records, indent=2)
        )

    def test_task2_address(self, containers_ready, tmp_path):
        """Task 2: SQL MCP agent on 2 address-domain samples."""
        output_dir = tmp_path / "task2"
        output_dir.mkdir()

        result = _run_benchmark(task_id=2, output_dir=output_dir)

        assert result.returncode == 0, (
            f"benchmark_runner.py exited with code {result.returncode} "
            f"(see output above)"
        )

        records = _assert_output(output_dir)

        successful = [r for r in records if r["status"] == "success"]
        assert len(successful) >= 1, (
            "Expected at least 1 successful record for task 2.\n"
            + json.dumps(records, indent=2)
        )
