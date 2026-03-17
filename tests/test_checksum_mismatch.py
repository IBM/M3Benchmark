#!/usr/bin/env python3
"""
Integration test: checksum validation catches a wrong domain.

This test connects to a real capability container (requires `docker compose up -d`),
fetches the live tool list for one domain, then tries to verify those tools against
a *different* domain's stored checksum. The mismatch must be detected and raise a
ValueError.

It also verifies the positive case — that the correct domain passes cleanly.

Usage:
    # Run directly
    python tests/test_checksum_mismatch.py

    # Or via pytest
    pytest tests/test_checksum_mismatch.py -v

Requires:
    docker compose up -d       # capability containers must be running
    make download              # benchmark data must be present
"""


import asyncio
import os
import shutil
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ── project root on path ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from environment.tool_checksums import verify_checksum  # noqa: E402

# ── constants ────────────────────────────────────────────────────────────────
# We connect to capability 2, domain "hockey" — then try to verify those tools
# against "address". The tool sets are different, so the checksum must fail.
CAPABILITY_ID  = 2
CONNECT_DOMAIN = "hockey"   # domain we actually connect to (real tools)
WRONG_DOMAIN   = "address"  # domain we claim when verifying (should mismatch)
CONTAINER      = "capability_2_dashboard_apis"
CONTAINER_CMD  = ["python", "/app/mcp_dispatch.py"]


# ── helpers ──────────────────────────────────────────────────────────────────

def _get_runtime() -> str:
    for rt in ("docker", "podman"):
        if shutil.which(rt):
            return rt
    pytest.skip("Neither docker nor podman found — skipping integration test.")


def _make_params(domain: str) -> StdioServerParameters:
    runtime = _get_runtime()
    args = [
        "exec", "-i",
        "-e", f"MCP_DOMAIN={domain}",
        "-e", "MCP_DB_ROOT=/app/db",
        "-e", f"CAPABILITY_ID={CAPABILITY_ID}",
        CONTAINER,
        *CONTAINER_CMD,
    ]
    return StdioServerParameters(command=runtime, args=args, env=None)


async def _fetch_tools(domain: str):
    """Connect to the container for *domain* and return the raw MCP tool list."""
    params = _make_params(domain)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            resp = await session.list_tools()
            return resp.tools


# ── tests ────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def hockey_tools():
    """Fetch the live tool list for the 'hockey' domain once per test session."""
    try:
        tools = asyncio.run(_fetch_tools(CONNECT_DOMAIN))
    except Exception as exc:
        pytest.skip(f"Could not connect to container '{CONTAINER}': {exc}")
    return tools


def test_correct_domain_passes(hockey_tools):
    """Verifying the real domain against its own checksum must succeed."""
    os.environ["MCP_VERIFY_CHECKSUMS"] = "1"
    try:
        # Should not raise
        verify_checksum(CAPABILITY_ID, CONNECT_DOMAIN, hockey_tools)
    finally:
        del os.environ["MCP_VERIFY_CHECKSUMS"]


def test_wrong_domain_raises(hockey_tools):
    """Verifying hockey tools against 'address' checksum must raise ValueError.

    This is the core guard: if MCP_DOMAIN is misconfigured the benchmark would
    silently evaluate the agent on the wrong tool set. The checksum catches it.
    """
    os.environ["MCP_VERIFY_CHECKSUMS"] = "1"
    try:
        with pytest.raises(ValueError, match="checksum mismatch"):
            verify_checksum(CAPABILITY_ID, WRONG_DOMAIN, hockey_tools)
    finally:
        del os.environ["MCP_VERIFY_CHECKSUMS"]


def test_disabled_by_default_no_error(hockey_tools):
    """Without MCP_VERIFY_CHECKSUMS set, even a wrong domain must not raise."""
    os.environ.pop("MCP_VERIFY_CHECKSUMS", None)
    # Should not raise regardless of domain mismatch
    verify_checksum(CAPABILITY_ID, WRONG_DOMAIN, hockey_tools)


# ── standalone entry point ────────────────────────────────────────────────────

def _run_standalone():
    print(f"Fetching tools from container '{CONTAINER}' (domain={CONNECT_DOMAIN!r}) ...")
    try:
        tools = asyncio.run(_fetch_tools(CONNECT_DOMAIN))
    except Exception as exc:
        print(f"SKIP — could not connect to container: {exc}")
        sys.exit(0)

    print(f"  Got {len(tools)} tool(s): {[t.name for t in tools[:5]]} ...")

    # ── positive: correct domain ──────────────────────────────────────────────
    os.environ["MCP_VERIFY_CHECKSUMS"] = "1"
    print(f"\n[1] Verifying tools against correct domain ({CONNECT_DOMAIN!r}) ...")
    verify_checksum(CAPABILITY_ID, CONNECT_DOMAIN, tools)
    print("    PASS — checksum matched.")

    # ── negative: wrong domain ────────────────────────────────────────────────
    print(f"\n[2] Verifying same tools against wrong domain ({WRONG_DOMAIN!r}) ...")
    try:
        verify_checksum(CAPABILITY_ID, WRONG_DOMAIN, tools)
        print("    FAIL — expected ValueError but none was raised!")
        sys.exit(1)
    except ValueError as exc:
        print(f"    PASS — ValueError raised as expected:\n")
        for line in str(exc).splitlines():
            print(f"      {line}")

    # ── negative: verification disabled ──────────────────────────────────────
    del os.environ["MCP_VERIFY_CHECKSUMS"]
    print(f"\n[3] Verifying wrong domain with MCP_VERIFY_CHECKSUMS unset ...")
    verify_checksum(CAPABILITY_ID, WRONG_DOMAIN, tools)
    print("    PASS — no error raised when verification is disabled.")

    print("\nAll checks passed.")


if __name__ == "__main__":
    _run_standalone()
