"""End-to-end tests for the benchmark pipeline.

Tests the full pipeline:
  1. Download data from HuggingFace (incremental sync)
  2. Start Docker containers
  3. Run benchmark_runner.py for capabilities 1, 2, 3, and 4
  4. Validate the output JSON files

Provider selection:
  Set E2E_PROVIDER to force a specific provider, or let auto-detection pick
  the first available credentials (priority: RITS > WatsonX > OpenAI):
  - RITS:      RITS_API_KEY
  - WatsonX:   WATSONX_APIKEY + (WATSONX_PROJECT_ID or WATSONX_SPACE_ID)
  - OpenAI:    OPENAI_API_KEY
  - LiteLLM:   LITELLM_API_KEY + LITELLM_BASE_URL  (requires E2E_PROVIDER=litellm)
  - Anthropic: ANTHROPIC_API_KEY                    (requires E2E_PROVIDER=anthropic)

Requirements:
    HF_TOKEN              - HuggingFace token for data download
    One of the above provider key sets
    Docker                - Available on PATH

Usage:
    # Install setup dependencies first (needed for HuggingFace data download)
    pip install -e '.[init]'

    # Set env vars and run (RITS example)
    RITS_API_KEY=... HF_TOKEN=hf_... python -m pytest tests/e2e/test_benchmark_e2e.py -v -s

    # WatsonX
    WATSONX_APIKEY=... WATSONX_PROJECT_ID=... HF_TOKEN=hf_... \\
        python -m pytest tests/e2e/test_benchmark_e2e.py -v -s

    # OpenAI
    OPENAI_API_KEY=sk-... HF_TOKEN=hf_... python -m pytest tests/e2e/test_benchmark_e2e.py -v -s

    # Using a .env file (copy template_env → .env and fill in values)
    cp template_env .env
    export $(grep -v '^#' .env | xargs)
    python -m pytest tests/e2e/test_benchmark_e2e.py -v -s

    # Run only one capability test
    python -m pytest tests/e2e/test_benchmark_e2e.py::TestBenchmarkE2E::test_capability1_address -v -s
    python -m pytest tests/e2e/test_benchmark_e2e.py::TestBenchmarkE2E::test_capability2_address -v -s
    python -m pytest tests/e2e/test_benchmark_e2e.py::TestBenchmarkE2E::test_capability3_airline -v -s
    python -m pytest tests/e2e/test_benchmark_e2e.py::TestBenchmarkE2E::test_capability3_bpo -v -s
    python -m pytest tests/e2e/test_benchmark_e2e.py::TestBenchmarkE2E::test_capability4_address -v -s

    # Skip container setup (containers already running)
    E2E_SKIP_SETUP=1 RITS_API_KEY=... python -m pytest tests/e2e/test_benchmark_e2e.py -v -s

    # LiteLLM (requires E2E_PROVIDER to avoid being shadowed by auto-detection)
    E2E_SKIP_SETUP=1 E2E_PROVIDER=litellm LITELLM_API_KEY=... LITELLM_BASE_URL=... \\
        python -m pytest tests/e2e/test_benchmark_e2e.py -v -s

    # Anthropic
    E2E_SKIP_SETUP=1 E2E_PROVIDER=anthropic ANTHROPIC_API_KEY=... \\
        python -m pytest tests/e2e/test_benchmark_e2e.py -v -s

    # Or use make targets (handle credential checks automatically):
    #   make e2e-quick            — OpenAI
    #   make e2e-quick-rits       — RITS
    #   make e2e-quick-watsonx    — WatsonX
    #   make e2e-quick-litellm    — LiteLLM
    #   make e2e-quick-anthropic  — Anthropic

Notes:
    - Containers are started once (module-scoped fixture) and shared across all tests.
    - Each capability writes to its own isolated output directory.
    - Tests fail immediately with a clear error if no provider keys are found.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def _detect_provider() -> Tuple[str, Optional[str]]:
    """Return (provider, model) based on E2E_PROVIDER or credential auto-detection.

    If E2E_PROVIDER is set, that provider is used directly (the Makefile
    e2e-quick-* targets set this).  Otherwise the first available credentials
    win (priority: RITS > WatsonX > OpenAI).

    Returns:
        Tuple of (provider_name, model_or_None).

    Raises:
        pytest.fail if no credentials are found.
    """
    forced = os.environ.get("E2E_PROVIDER", "").strip().lower()

    if forced == "litellm":
        model = os.environ.get("LITELLM_MODEL", None)
        return "litellm", model

    if forced == "anthropic":
        model = os.environ.get("ANTHROPIC_MODEL", None)
        return "anthropic", model

    # Auto-detect: RITS > WatsonX > OpenAI
    # 1. RITS
    if os.environ.get("RITS_API_KEY"):
        model = os.environ.get("RITS_MODEL", "llama-3-3-70b-instruct")
        return "rits", model

    # 2. WatsonX — needs key + at least one of project_id / space_id
    wx_key = os.environ.get("WATSONX_APIKEY")
    wx_project = os.environ.get("WATSONX_PROJECT_ID")
    wx_space = os.environ.get("WATSONX_SPACE_ID")
    if wx_key and (wx_project or wx_space):
        model = os.environ.get("WATSONX_MODEL", "openai/gpt-oss-120b")
        return "watsonx", model

    # 3. OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        model = os.environ.get("OPENAI_MODEL", None)  # use benchmark_runner default
        return "openai", model

    pytest.fail(
        "No LLM provider credentials found. Set one of:\n"
        "  RITS_API_KEY\n"
        "  WATSONX_APIKEY + WATSONX_PROJECT_ID (or WATSONX_SPACE_ID)\n"
        "  OPENAI_API_KEY\n"
        "  LITELLM_API_KEY + LITELLM_BASE_URL  (with E2E_PROVIDER=litellm)\n"
        "  ANTHROPIC_API_KEY                    (with E2E_PROVIDER=anthropic)"
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def containers_ready():
    """Download benchmark data and start Docker containers.

    Runs once for the whole module; tears down containers on exit.
    Errors immediately if no provider credentials are set.
    """
    # Will call pytest.fail() if no provider is available
    _detect_provider()

    if os.environ.get("E2E_SKIP_SETUP"):
        yield  # containers assumed to be already running
        return

    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        pytest.fail(
            "Required environment variable not set: HF_TOKEN\n"
            "(or set E2E_SKIP_SETUP=1 to skip data download + container start)"
        )

    sys.path.insert(0, str(PROJECT_ROOT))
    from benchmark_setup import download_data, start_containers

    download_data()
    start_containers()
    yield
    # stop_containers()  # intentionally left running for faster re-runs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_benchmark(
    capability_id: int,
    output_dir: Path,
    domain: str = "address",
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """Invoke benchmark_runner.py as a subprocess, streaming output to the terminal."""
    if provider is None:
        provider, model = _detect_provider()

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "benchmark_runner.py"),
        "--capability_id", str(capability_id),
        "--domain", domain,
        "--provider", provider,
        "--max-samples-per-domain", "2",
        "--top-k-tools", "100",
        "--output", str(output_dir),
    ]
    if model:
        cmd += ["--model", model]

    print(f"\n$ {' '.join(str(c) for c in cmd)}\n", flush=True)
    return subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        timeout=300,  # 5 minutes — generous for 2 samples
    )


def _list_tools(capability_id: int, domain: str = "address") -> subprocess.CompletedProcess:
    """Invoke benchmark_runner.py --list-tools and capture its output."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "benchmark_runner.py"),
        "--capability_id", str(capability_id),
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
        for key in ("uuid", "domain", "status", "duration_s", "output"):
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

        # output is a non-empty list of turn records
        out_list = record["output"]
        assert isinstance(out_list, list) and len(out_list) >= 1, (
            f"{ctx}: 'output' must be a non-empty list"
        )

        for j, turn in enumerate(out_list):
            turn_ctx = f"{ctx}.output[{j}]"
            for key in ("query", "answer", "sequence"):
                assert key in turn, f"{turn_ctx}: missing key '{key}'"
            assert isinstance(turn["sequence"], dict), (
                f"{turn_ctx}: 'sequence' must be a dict"
            )
            assert "tool_call" in turn["sequence"], (
                f"{turn_ctx}: 'sequence' must contain 'tool_call'"
            )
            assert isinstance(turn["sequence"]["tool_call"], list), (
                f"{turn_ctx}: 'sequence.tool_call' must be a list"
            )

    return records


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBenchmarkE2E:
    """Full end-to-end tests: containers → runner → output validation."""

    def test_capability1_address(self, tmp_path):
        """Capability 1: slot-filling agent on 2 address-domain samples."""
        output_dir = tmp_path / "cap1"
        output_dir.mkdir()

        result = _run_benchmark(capability_id=1, output_dir=output_dir, domain="address")

        assert result.returncode == 0, (
            f"benchmark_runner.py exited with code {result.returncode} (see output above)"
        )
        records = _assert_output(output_dir, domain="address")
        successful = [r for r in records if r["status"] == "success"]
        assert len(successful) >= 1, (
            "Expected at least 1 successful record for capability 1.\n"
            + json.dumps(records, indent=2)
        )

    def test_capability2_movie(self, tmp_path):
        """Capability 2: SQL MCP agent on 2 movie-domain samples."""
        output_dir = tmp_path / "cap2"
        output_dir.mkdir()

        result = _run_benchmark(capability_id=2, output_dir=output_dir, domain="movie")

        assert result.returncode == 0, (
            f"benchmark_runner.py exited with code {result.returncode} (see output above)"
        )
        records = _assert_output(output_dir, domain="movie")
        successful = [r for r in records if r["status"] == "success"]
        assert len(successful) >= 1, (
            "Expected at least 1 successful record for capability 2.\n"
            + json.dumps(records, indent=2)
        )

    def test_capability3_airline(self, tmp_path):
        """Capability 3: M3 REST MCP agent on 2 airline-domain samples.

        Uses capability_3_multihop_reasoning container via bpo_router.py. Since "airline"
        is an M3 REST domain, the router exec's into the M3 REST MCP server.
        """
        output_dir = tmp_path / "cap3_airline"
        output_dir.mkdir()

        result = _run_benchmark(capability_id=3, output_dir=output_dir, domain="airline")

        assert result.returncode == 0, (
            f"benchmark_runner.py exited with code {result.returncode} (see output above)"
        )
        records = _assert_output(output_dir, domain="airline")
        successful = [r for r in records if r["status"] == "success"]
        assert len(successful) >= 1, (
            "Expected at least 1 successful record for capability 3 (airline).\n"
            + json.dumps(records, indent=2)
        )

    def test_capability3_hockey(self, tmp_path):
        """Capability 3: multihop reasoning agent on 2 hockey-domain samples."""
        output_dir = tmp_path / "cap3_hockey"
        output_dir.mkdir()

        result = _run_benchmark(capability_id=3, output_dir=output_dir, domain="hockey")

        assert result.returncode == 0, (
            f"benchmark_runner.py exited with code {result.returncode} (see output above)"
        )
        records = _assert_output(output_dir, domain="hockey")
        successful = [r for r in records if r["status"] == "success"]
        assert len(successful) >= 1, (
            "Expected at least 1 successful record for capability 3 (hockey).\n"
            + json.dumps(records, indent=2)
        )

    def test_capability4_combined_tools(self):
        """Capability 4: combined MCP server exposes address M3 REST tools + retriever tools.

        For MCP_DOMAIN=address the expected tool set is:
          - M3 REST:  all /v1/address/* tools only  (~40 tools)
          - Retriever: query_address + negative-domain retriever tools

        Verifies four things:
        1. query_address is present   — primary retriever tool included.
        2. query_hockey is present    — negative-domain retriever tool included
                                        (address negatives: hockey, law_episode,
                                         trains, public_review_platform).
        3. Total tool count > 1       — M3 REST tools are also included.
        4. query_airline is absent    — unrelated domain not leaking through.
        """
        result = _list_tools(capability_id=4, domain="address")

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
        assert "query_hockey" in result.stdout, (
            "Expected 'query_hockey' (a negative domain for 'address' per "
            "domain_negatives.json) to be exposed, but it was not found.\n"
            f"Output:\n{result.stdout}"
        )

        # 3. M3 REST tools must be present (combined count >> retriever-only)
        assert total_tools > 1, (
            f"Expected more than 1 tool (got {total_tools}). "
            "The combined server should expose M3 REST tools AND retriever tools."
        )

        # 4. Unrelated domains must not appear
        assert "query_airline" not in result.stdout, (
            "Found 'query_airline' in capability 4 address output — "
            "domain filtering is broken; unrelated tools are leaking through."
        )

    def test_capability4_address(self, tmp_path):
        """Capability 4: combined MCP server (M3 REST + retriever) on 2 address samples.

        Uses capability_4_multiturn container which merges tools from both FastAPI
        servers so the agent can use SQL/REST tools and semantic search in the same
        session.
        """
        output_dir = tmp_path / "cap4"
        output_dir.mkdir()

        result = _run_benchmark(capability_id=4, output_dir=output_dir, domain="address")

        assert result.returncode == 0, (
            f"benchmark_runner.py exited with code {result.returncode} (see output above)"
        )
        records = _assert_output(output_dir, domain="address")
        successful = [r for r in records if r["status"] == "success"]
        assert len(successful) >= 1, (
            "Expected at least 1 successful record for capability 4.\n"
            + json.dumps(records, indent=2)
        )
