"""End-to-end tests for the M3 benchmark pipeline.

Tests the full pipeline:
  1. Download data from HuggingFace (incremental sync)
  2. Start Docker containers
  3. Run benchmark_runner.py for task 1, task 2, task 3, and task 5
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
    python -m pytest tests/e2e/test_benchmark_e2e.py::TestBenchmarkE2E::test_task3_airline -v -s
    python -m pytest tests/e2e/test_benchmark_e2e.py::TestBenchmarkE2E::test_task3_bpo -v -s
    python -m pytest tests/e2e/test_benchmark_e2e.py::TestBenchmarkE2E::test_task5_address -v -s

    # Run the entire test suite in one go
    python -m pytest tests/e2e/test_benchmark_e2e.py -v -s

Notes:
    - Containers are started once (module-scoped fixture) and shared across all tests.
    - Each task writes to its own isolated output directory.
    - Tests fail immediately with a clear error if required env vars are absent.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def containers_ready():
    """Download benchmark data and start Docker containers.

    Runs once for the whole module; tears down containers on exit.
    Errors immediately if required environment variables are not set.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.fail("Required environment variable not set: OPENAI_API_KEY")

    if os.environ.get("E2E_SKIP_SETUP"):
        yield  # containers assumed to be already running
        return

    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        pytest.fail("Required environment variable not set: HF_TOKEN (or set E2E_SKIP_SETUP=1 to skip setup)")

    sys.path.insert(0, str(PROJECT_ROOT))
    from m3_setup import download_data, start_containers, stop_containers

    download_data()
    start_containers()
    yield
    # stop_containers()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_benchmark(task_id: int, output_dir: Path, domain: str = "address") -> subprocess.CompletedProcess:
    """Invoke benchmark_runner.py as a subprocess, streaming output to the terminal."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "benchmark_runner.py"),
        "--m3_task_id", str(task_id),
        "--domain", domain,
        "--provider", "openai",
        "--max-samples-per-domain", "2",
        "--top-k-tools", "100",
        "--output", str(output_dir),
    ]
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n", flush=True)
    return subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        timeout=300,  # 5 minutes — generous for 2 samples
    )


def _list_tools(task_id: int, domain: str = "address") -> subprocess.CompletedProcess:
    """Invoke benchmark_runner.py --list-tools and capture its output."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "benchmark_runner.py"),
        "--m3_task_id", str(task_id),
        "--domain", domain,
        "--list-tools",
    ]
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n", flush=True)
    return subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
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

    def test_task1_address(self, tmp_path):
        """Task 1: slot-filling agent on 2 address-domain samples."""
        output_dir = tmp_path / "task1"
        output_dir.mkdir()

        result = _run_benchmark(task_id=1, output_dir=output_dir, domain="address")

        assert result.returncode == 0, (
            f"benchmark_runner.py exited with code {result.returncode} (see output above)"
        )
        records = _assert_output(output_dir, domain="address")
        successful = [r for r in records if r["status"] == "success"]
        assert len(successful) >= 1, (
            "Expected at least 1 successful record for task 1.\n"
            + json.dumps(records, indent=2)
        )

    def test_task2_address(self, tmp_path):
        """Task 2: SQL MCP agent on 2 address-domain samples."""
        output_dir = tmp_path / "task2"
        output_dir.mkdir()

        result = _run_benchmark(task_id=2, output_dir=output_dir, domain="address")

        assert result.returncode == 0, (
            f"benchmark_runner.py exited with code {result.returncode} (see output above)"
        )
        records = _assert_output(output_dir, domain="address")
        successful = [r for r in records if r["status"] == "success"]
        assert len(successful) >= 1, (
            "Expected at least 1 successful record for task 2.\n"
            + json.dumps(records, indent=2)
        )

    def test_task3_airline(self, tmp_path):
        """Task 3: M3 REST MCP agent on 2 airline-domain samples.

        Uses task_3_m3_environ container via task3_router.py. Since "airline"
        is an M3 REST domain, the router exec's into /app/m3-rest/mcp_server.py.
        """
        output_dir = tmp_path / "task3_airline"
        output_dir.mkdir()

        result = _run_benchmark(task_id=3, output_dir=output_dir, domain="airline")

        assert result.returncode == 0, (
            f"benchmark_runner.py exited with code {result.returncode} (see output above)"
        )
        records = _assert_output(output_dir, domain="airline")
        successful = [r for r in records if r["status"] == "success"]
        assert len(successful) >= 1, (
            "Expected at least 1 successful record for task 3 (airline).\n"
            + json.dumps(records, indent=2)
        )

    def test_task3_bpo(self, tmp_path):
        """Task 3: BPO MCP agent on 2 BPO-domain samples.

        Uses task_3_m3_environ container via task3_router.py. Since "bpo"
        is the BPO domain, the router exec's into /app/apis/bpo/mcp/server.py.
        """
        output_dir = tmp_path / "task3_bpo"
        output_dir.mkdir()

        result = _run_benchmark(task_id=3, output_dir=output_dir, domain="bpo")

        assert result.returncode == 0, (
            f"benchmark_runner.py exited with code {result.returncode} (see output above)"
        )
        records = _assert_output(output_dir, domain="bpo")
        successful = [r for r in records if r["status"] == "success"]
        assert len(successful) >= 1, (
            "Expected at least 1 successful record for task 3 (bpo).\n"
            + json.dumps(records, indent=2)
        )

    def test_task5_combined_tools(self):
        """Task 5: combined MCP server exposes address M3 REST tools + retriever
        tools for address and its negative domains.

        For MCP_DOMAIN=address the expected tool set is:
          - M3 REST:  all /v1/address/* tools only  (~40 tools)
          - Retriever: query_address, query_olympics, query_card_games,
                       query_legislator, query_craftbeer  (5 tools from
                       domain_negatives.json["address"])

        Verifies four things:
        1. query_address is present   — primary retriever tool included.
        2. query_olympics is present  — negative-domain retriever tool included.
        3. Total tool count > 1       — M3 REST tools are also included.
        4. query_hockey is absent     — unrelated domain not leaking through.
        """
        result = _list_tools(task_id=5, domain="address")

        print(result.stdout, flush=True)
        if result.stderr:
            print(result.stderr, flush=True)

        assert result.returncode == 0, (
            f"--list-tools exited with code {result.returncode}\n{result.stderr}"
        )

        # Parse "  Total tools: N" from the output
        total_tools = None
        for line in result.stdout.splitlines():
            if "Total tools:" in line:
                try:
                    total_tools = int(line.split("Total tools:")[-1].strip())
                except ValueError:
                    pass
                break

        assert total_tools is not None, (
            f"Could not find 'Total tools:' in --list-tools output:\n{result.stdout}"
        )

        # 1. Primary retriever tool must be present
        assert "query_address" in result.stdout, (
            "Expected 'query_address' tool from the retriever backend, "
            f"but it was not found in --list-tools output:\n{result.stdout}"
        )

        # 2. Negative-domain retriever tools must also be present
        assert "query_olympics" in result.stdout, (
            "Expected 'query_olympics' (a negative domain for 'address' per "
            "domain_negatives.json) to be exposed, but it was not found.\n"
            f"Output:\n{result.stdout}"
        )

        # 3. M3 REST tools must be present (combined count >> retriever-only)
        assert total_tools > 1, (
            f"Expected more than 1 tool (got {total_tools}). "
            "The combined server should expose M3 REST tools AND retriever tools."
        )

        # 4. Unrelated domains must not appear
        assert "query_hockey" not in result.stdout, (
            "Found 'query_hockey' in task 5 address output — "
            "domain filtering is broken; unrelated tools are leaking through."
        )

    def test_task5_address(self, tmp_path):
        """Task 5: combined MCP server (M3 REST + retriever) on 2 address samples.

        Uses task5_mcp_server.py which merges tools from both FastAPI servers
        running in task_5_m3_environ so the agent can use SQL/REST tools and
        semantic search in the same session.
        """
        output_dir = tmp_path / "task5"
        output_dir.mkdir()

        result = _run_benchmark(task_id=5, output_dir=output_dir, domain="address")

        assert result.returncode == 0, (
            f"benchmark_runner.py exited with code {result.returncode} (see output above)"
        )
        records = _assert_output(output_dir, domain="address")
        successful = [r for r in records if r["status"] == "success"]
        assert len(successful) >= 1, (
            "Expected at least 1 successful record for task 5.\n"
            + json.dumps(records, indent=2)
        )
