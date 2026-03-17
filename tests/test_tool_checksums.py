"""Unit tests for tool checksum utilities.

Tests verify:
  1. Checksum computation is stable and order-independent.
  2. Checksums differ across domains/tool-sets.
  3. verify_checksum passes when expected == actual.
  4. verify_checksum raises ValueError on mismatch (wrong domain guard).
  5. verify_checksum warns but does NOT raise when no checksum is registered.
  6. Pydantic-model inputSchema is handled correctly.
  7. The committed tool_checksums.json is structurally valid and complete.

These tests are fully self-contained — no Docker containers required.

Usage:
    python -m pytest tests/test_tool_checksums.py -v
"""

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from environment.tool_checksums import (
    compute_tool_checksum,
    load_checksums,
    verify_checksum,
    _verification_enabled,
)


# ---------------------------------------------------------------------------
# Helpers — lightweight stand-ins for mcp.types.Tool
# ---------------------------------------------------------------------------

def _make_tool(name: str, schema: Dict[str, Any]) -> SimpleNamespace:
    """Return a minimal object that mimics mcp.types.Tool."""
    t = SimpleNamespace()
    t.name = name
    t.inputSchema = schema
    return t


def _make_tools(names: List[str]) -> List[SimpleNamespace]:
    """Create a list of tools with simple schemas, one per name."""
    return [
        _make_tool(
            name,
            {
                "type": "object",
                "properties": {"q": {"type": "string", "description": "query"}},
                "required": ["q"],
            },
        )
        for name in names
    ]


def _write_checksums(data: Dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# compute_tool_checksum
# ---------------------------------------------------------------------------

class TestComputeToolChecksum:
    def test_returns_64_char_hex(self):
        tools = _make_tools(["tool_a"])
        result = compute_tool_checksum(tools)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_stable_across_calls(self):
        tools = _make_tools(["tool_a", "tool_b"])
        assert compute_tool_checksum(tools) == compute_tool_checksum(tools)

    def test_order_independent(self):
        tools_ab = _make_tools(["tool_a", "tool_b"])
        tools_ba = _make_tools(["tool_b", "tool_a"])
        assert compute_tool_checksum(tools_ab) == compute_tool_checksum(tools_ba)

    def test_different_names_differ(self):
        tools_a = _make_tools(["address_tool"])
        tools_b = _make_tools(["airline_tool"])
        assert compute_tool_checksum(tools_a) != compute_tool_checksum(tools_b)

    def test_different_schemas_differ(self):
        tool_v1 = _make_tool("t", {"type": "object", "properties": {}, "required": []})
        tool_v2 = _make_tool(
            "t",
            {
                "type": "object",
                "properties": {"extra": {"type": "integer"}},
                "required": ["extra"],
            },
        )
        assert compute_tool_checksum([tool_v1]) != compute_tool_checksum([tool_v2])

    def test_empty_list(self):
        result = compute_tool_checksum([])
        assert len(result) == 64

    def test_dict_input(self):
        """Accepts plain dicts as well as objects with attributes."""
        tool_obj = _make_tools(["t"])[0]
        tool_dict = {"name": "t", "inputSchema": tool_obj.inputSchema}
        assert compute_tool_checksum([tool_obj]) == compute_tool_checksum([tool_dict])

    def test_pydantic_v2_schema(self):
        """inputSchema that is a Pydantic v2 model is handled via model_dump()."""
        mock_schema = MagicMock()
        mock_schema.model_dump.return_value = {"type": "object", "properties": {}}
        # Ensure hasattr(..., "dict") returns False so model_dump branch is taken
        del mock_schema.dict
        tool = SimpleNamespace(name="pydantic_tool", inputSchema=mock_schema)
        result = compute_tool_checksum([tool])
        assert len(result) == 64
        mock_schema.model_dump.assert_called_once()


# ---------------------------------------------------------------------------
# load_checksums
# ---------------------------------------------------------------------------

class TestLoadChecksums:
    def test_returns_empty_dict_when_missing(self, tmp_path):
        result = load_checksums(tmp_path / "nonexistent.json")
        assert result == {}

    def test_loads_valid_file(self, tmp_path):
        data = {"2": {"address": "abc123"}, "4": {"address": "def456"}}
        p = tmp_path / "checksums.json"
        _write_checksums(data, p)
        assert load_checksums(p) == data

    def test_ignores_comment_key(self, tmp_path):
        """_comment key is present in the file but doesn't break loading."""
        data = {"_comment": "ignore me", "2": {"address": "abc"}}
        p = tmp_path / "checksums.json"
        _write_checksums(data, p)
        result = load_checksums(p)
        assert result["2"] == {"address": "abc"}


# ---------------------------------------------------------------------------
# verify_checksum — flag control
# ---------------------------------------------------------------------------

class TestVerificationFlag:
    """MCP_VERIFY_CHECKSUMS env var controls whether verification runs."""

    def test_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MCP_VERIFY_CHECKSUMS", None)
            assert _verification_enabled() is False

    def test_enabled_with_1(self):
        with patch.dict(os.environ, {"MCP_VERIFY_CHECKSUMS": "1"}):
            assert _verification_enabled() is True

    def test_enabled_with_true(self):
        with patch.dict(os.environ, {"MCP_VERIFY_CHECKSUMS": "true"}):
            assert _verification_enabled() is True

    def test_enabled_case_insensitive(self):
        with patch.dict(os.environ, {"MCP_VERIFY_CHECKSUMS": "TRUE"}):
            assert _verification_enabled() is True

    def test_disabled_with_0(self):
        with patch.dict(os.environ, {"MCP_VERIFY_CHECKSUMS": "0"}):
            assert _verification_enabled() is False

    def test_no_error_when_disabled_even_with_mismatch(self, tmp_path):
        """When flag is off, verify_checksum is a no-op regardless of stored checksums."""
        tools_correct = _make_tools(["tool_x"])
        tools_wrong = _make_tools(["tool_y"])
        checksum = compute_tool_checksum(tools_correct)
        p = tmp_path / "cs.json"
        _write_checksums({"2": {"address": checksum}}, p)

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MCP_VERIFY_CHECKSUMS", None)
            # Should NOT raise even though tools don't match
            verify_checksum(2, "address", tools_wrong, checksums_path=p)


# ---------------------------------------------------------------------------
# verify_checksum — matching and mismatch
# ---------------------------------------------------------------------------

class TestVerifyChecksum:
    @pytest.fixture(autouse=True)
    def _enable(self, monkeypatch):
        monkeypatch.setenv("MCP_VERIFY_CHECKSUMS", "1")

    def _write_with_real_checksum(
        self, tmp_path: Path, capability_id: int, domain: str, tools
    ) -> Path:
        """Write a checksums file with the real checksum for tools."""
        checksum = compute_tool_checksum(tools)
        data = {str(capability_id): {domain: checksum}}
        p = tmp_path / "tool_checksums.json"
        _write_checksums(data, p)
        return p

    # --- happy path ---

    def test_passes_when_tools_match(self, tmp_path):
        tools = _make_tools(["get_address_streets", "filter_address"])
        p = self._write_with_real_checksum(tmp_path, capability_id=2, domain="address", tools=tools)
        # Should not raise
        verify_checksum(2, "address", tools, checksums_path=p)

    # --- mismatch (wrong domain) ---

    def test_raises_on_wrong_domain_tools(self, tmp_path):
        address_tools = _make_tools(["get_address_streets", "filter_address"])
        airline_tools = _make_tools(["get_airline_routes", "filter_airline"])

        # Store checksum for "address"
        p = self._write_with_real_checksum(tmp_path, 2, "address", address_tools)

        # Client accidentally connected with MCP_DOMAIN=airline, so it gets
        # airline tools, but is asserting against "address" checksum.
        with pytest.raises(ValueError, match="checksum mismatch"):
            verify_checksum(2, "address", airline_tools, checksums_path=p)

    def test_error_message_contains_capability_and_domain(self, tmp_path):
        tools_correct = _make_tools(["tool_x"])
        tools_wrong = _make_tools(["tool_y"])
        p = self._write_with_real_checksum(tmp_path, 3, "bpo", tools_correct)

        with pytest.raises(ValueError) as exc_info:
            verify_checksum(3, "bpo", tools_wrong, checksums_path=p)

        msg = str(exc_info.value)
        assert "capability 3" in msg
        assert "'bpo'" in msg
        assert "Expected" in msg
        assert "Got" in msg

    # --- missing checksums (warn, don't error) ---

    def test_no_error_when_capability_not_registered(self, tmp_path, caplog):
        tools = _make_tools(["t"])
        p = tmp_path / "empty.json"
        _write_checksums({}, p)

        with caplog.at_level(logging.WARNING, logger="environment.tool_checksums"):
            verify_checksum(2, "address", tools, checksums_path=p)  # must not raise

        assert any("No checksums registered for capability" in r.message for r in caplog.records)

    def test_no_error_when_domain_not_registered(self, tmp_path, caplog):
        tools = _make_tools(["t"])
        p = tmp_path / "partial.json"
        _write_checksums({"2": {"hockey": "somehash"}}, p)

        with caplog.at_level(logging.WARNING, logger="environment.tool_checksums"):
            verify_checksum(2, "address", tools, checksums_path=p)  # must not raise

        assert any("No checksum registered" in r.message for r in caplog.records)

    def test_no_error_when_file_missing(self, tmp_path, caplog):
        tools = _make_tools(["t"])

        with caplog.at_level(logging.WARNING, logger="environment.tool_checksums"):
            verify_checksum(2, "address", tools, checksums_path=tmp_path / "missing.json")

        # Should warn (no registered capability) but not raise
        assert any("No checksums registered" in r.message for r in caplog.records)

    # --- checksums are independent per capability ---

    def test_same_tools_different_capability_independent(self, tmp_path):
        """Tasks/capabilities are looked up independently in the checksums dict."""
        tools = _make_tools(["shared_tool"])
        checksum = compute_tool_checksum(tools)
        data = {
            "2": {"address": checksum},
            "4": {"address": "different_checksum_here"},
        }
        p = tmp_path / "cs.json"
        _write_checksums(data, p)

        # Capability 2 address matches
        verify_checksum(2, "address", tools, checksums_path=p)

        # Capability 4 address does NOT match (different stored value)
        with pytest.raises(ValueError):
            verify_checksum(4, "address", tools, checksums_path=p)


# ---------------------------------------------------------------------------
# Committed tool_checksums.json — structural and content validation
# ---------------------------------------------------------------------------

COMMITTED_CHECKSUMS_PATH = Path(__file__).parent.parent / "tool_checksums.json"

# Expected capabilities and a representative sample of domains per capability.
# These must be present in the committed file.
_EXPECTED_CAP2_DOMAINS = {
    "california_schools", "card_games", "chicago_crime", "debit_card_specializing",
    "european_football_2", "financial", "formula_1", "movie", "movie_3",
    "movielens", "movies_4", "public_review_platform", "simpson_episodes",
    "superhero", "thrombosis_prediction", "toxicology", "video_games",
}

_EXPECTED_CAP3_DOMAINS = {
    "address", "airline", "app_store", "beer_factory", "bike_share_1",
    "books", "cars", "chicago_crime", "codebase_comments", "coinmarketcap",
    "computer_student", "cookbook", "european_football_1", "food_inspection",
    "genes", "hockey", "ice_hockey_draft", "law_episode", "menu", "mondial_geo",
    "movie", "movie_3", "movielens", "movies_4", "olympics",
    "professional_basketball", "public_review_platform", "restaurant",
    "sales_in_weather", "shakespeare", "simpson_episodes", "soccer_2016",
    "student_loan", "talkingdata", "university", "video_games",
    "world_development_indicators",
}

_EXPECTED_CAP4_DOMAINS = {
    "address", "authors", "beer_factory", "bike_share_1", "book_publishing_company",
    "books", "chicago_crime", "citeseer", "codebase_comments", "coinmarketcap",
    "college_completion", "computer_student", "cookbook", "disney",
    "european_football_1", "food_inspection", "hockey", "ice_hockey_draft",
    "image_and_language", "law_episode", "menu", "mondial_geo", "movie",
    "movie_3", "movielens", "movies_4", "music_tracker", "olympics",
    "professional_basketball", "public_review_platform", "restaurant",
    "shakespeare", "simpson_episodes", "soccer_2016", "student_loan",
    "talkingdata", "trains", "university", "video_games", "world",
    "world_development_indicators",
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class TestCommittedChecksums:
    """Validate the committed tool_checksums.json without needing Docker."""

    @pytest.fixture(scope="class")
    def checksums(self):
        assert COMMITTED_CHECKSUMS_PATH.exists(), (
            f"tool_checksums.json not found at {COMMITTED_CHECKSUMS_PATH}. "
            "Run generate_checksums.py to create it."
        )
        with open(COMMITTED_CHECKSUMS_PATH) as f:
            return json.load(f)

    def test_file_is_valid_json(self):
        with open(COMMITTED_CHECKSUMS_PATH) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_all_supported_capabilities_present(self, checksums):
        for cap in ("2", "3", "4"):
            assert cap in checksums, f"capability '{cap}' missing from tool_checksums.json"

    def test_no_empty_capability(self, checksums):
        for cap in ("2", "3", "4"):
            assert checksums[cap], f"capability '{cap}' has no domains in tool_checksums.json"

    def test_all_checksums_are_sha256_hex(self, checksums):
        for cap, domains in checksums.items():
            if cap == "_comment":
                continue
            for domain, checksum in domains.items():
                assert _SHA256_RE.match(checksum), (
                    f"capability {cap}, domain '{domain}': "
                    f"checksum '{checksum}' is not a valid 64-char hex SHA-256"
                )

    def test_no_duplicate_checksums_within_capability(self, checksums):
        """Every domain within a capability should have a unique checksum.

        A duplicate strongly suggests a copy-paste error or that the same
        server responded for two different domains.
        """
        for cap, domains in checksums.items():
            if cap == "_comment":
                continue
            seen: dict[str, str] = {}
            for domain, checksum in domains.items():
                assert checksum not in seen, (
                    f"capability {cap}: domains '{seen[checksum]}' and '{domain}' "
                    f"share the same checksum ({checksum[:16]}…) — likely a generation error"
                )
                seen[checksum] = domain

    def test_capability_2_expected_domains_present(self, checksums):
        present = set(checksums["2"].keys())
        missing = _EXPECTED_CAP2_DOMAINS - present
        assert not missing, f"capability 2 missing domains: {sorted(missing)}"

    def test_capability_3_expected_domains_present(self, checksums):
        present = set(checksums["3"].keys())
        missing = _EXPECTED_CAP3_DOMAINS - present
        assert not missing, f"capability 3 missing domains: {sorted(missing)}"

    def test_capability_4_expected_domains_present(self, checksums):
        present = set(checksums["4"].keys())
        missing = _EXPECTED_CAP4_DOMAINS - present
        assert not missing, f"capability 4 missing domains: {sorted(missing)}"

    def test_shared_domains_have_independent_checksums(self, checksums):
        """Domains that appear in multiple capabilities should NOT share a checksum
        unless the tool sets are genuinely identical (e.g. same SQL tools reused).

        This test flags same-name/same-checksum pairs across capabilities so they
        can be reviewed, but does not hard-fail — the real tools may legitimately
        share a definition.
        """
        # Domains present in both cap 2 and cap 3 with the same checksum are
        # expected (same M3 REST server, same tools). Between cap 3 and cap 4
        # the retriever is added, so checksums should differ.
        cap3 = checksums.get("3", {})
        cap4 = checksums.get("4", {})
        shared = set(cap3.keys()) & set(cap4.keys())
        same = [d for d in shared if cap3[d] == cap4[d]]
        assert not same, (
            f"These domains have identical checksums in cap 3 and cap 4 "
            f"(cap 4 should include retriever tools): {sorted(same)}"
        )

    def test_verify_checksum_accepts_committed_value(self, checksums, monkeypatch):
        """verify_checksum should not raise when we pass the stored checksum back.

        We reconstruct a fake tool-list whose checksum equals the stored one —
        this validates that load_checksums + verify_checksum round-trip correctly
        using the real file, without needing Docker.
        """
        monkeypatch.setenv("MCP_VERIFY_CHECKSUMS", "1")

        # Pick the first domain from capability 2 as a representative sample.
        cap2 = checksums["2"]
        domain, expected_checksum = next(iter(cap2.items()))

        # Build a synthetic tool list that produces the same checksum.
        # We can't reverse a SHA-256, so instead we patch compute_tool_checksum
        # to return the stored value and confirm verify_checksum accepts it.
        with patch(
            "environment.tool_checksums.compute_tool_checksum",
            return_value=expected_checksum,
        ):
            verify_checksum(
                2, domain, _make_tools(["placeholder"]),
                checksums_path=COMMITTED_CHECKSUMS_PATH,
            )

    def test_verify_checksum_rejects_tampered_value(self, checksums, monkeypatch):
        """verify_checksum must raise ValueError if the checksum doesn't match
        the committed value — simulates a wrong-domain connection.
        """
        monkeypatch.setenv("MCP_VERIFY_CHECKSUMS", "1")
        cap2 = checksums["2"]
        domain = next(iter(cap2))

        with patch(
            "environment.tool_checksums.compute_tool_checksum",
            return_value="a" * 64,  # deliberately wrong checksum
        ):
            with pytest.raises(ValueError, match="checksum mismatch"):
                verify_checksum(
                    2, domain, _make_tools(["placeholder"]),
                    checksums_path=COMMITTED_CHECKSUMS_PATH,
                )
