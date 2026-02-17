"""Error-prone candidate source API variants for testing agent resilience.

Each function has a unique, plausible intent and embeds a specific error behavior.
Completely independent from original APIs — accesses DataLoader directly.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from .data_loader import get_data_loader

# Seeded RNG for reproducible probabilistic behavior
_rng = random.Random(42)

# Track call counts for rate-limiting behavior
_call_counts: Dict[str, int] = {}


def _check_requisition(requisition_id: str) -> Optional[Dict[str, Any]]:
    """Return error dict if requisition invalid, else None."""
    loader = get_data_loader()
    if not loader.is_valid_requisition(requisition_id):
        return {
            "error": "requisition_not_found",
            "message": f"Requisition {requisition_id} not found",
        }
    return None


# ── Test 28: Type mismatch — int instead of float ───────────────────────────

def get_source_sla_score(requisition_id: str, source_name: str = "Dice") -> Any:
    """Get the SLA score for a specific sourcing channel.

    Returns the SLA achievement score for the given source.

    ERROR BEHAVIOR: Returns int (e.g., 80) instead of float (e.g., 80.0).
    Tests type handling for numeric values.
    """
    err = _check_requisition(requisition_id)
    if err:
        return err

    loader = get_data_loader()
    data = loader.get_similar_requisitions(requisition_id)
    reviewed = data[(data["reviewed"]) & (data["source_name"] == source_name)]

    if len(reviewed) == 0:
        return {"error": "no_data", "message": f"No reviewed candidates from {source_name}"}

    sla_pct = int(round(reviewed["sla_met"].mean() * 100))
    return sla_pct  # Returns bare int instead of {"sla_score": 80.0}


# ── Test 29: Type mismatch — None instead of empty list ─────────────────────

def get_inactive_sources(requisition_id: str) -> Any:
    """Show any inactive sourcing channels with no candidates.

    Returns a list of sources that produced zero candidates.

    ERROR BEHAVIOR: Returns None instead of empty list when no
    inactive sources exist. Tests null handling.
    """
    err = _check_requisition(requisition_id)
    if err:
        return err

    loader = get_data_loader()
    data = loader.get_similar_requisitions(requisition_id)
    active_sources = set(data["source_name"].unique())

    # Use all sources from the full dataset as the reference set
    all_possible = set(loader.data["source_name"].unique())
    inactive = all_possible - active_sources

    if not inactive:
        return None  # Returns None instead of []

    return list(inactive)


# ── Test 30: HTTP 404 (probabilistic, 20% chance) ───────────────────────────

def get_candidate_pipeline_status(requisition_id: str) -> Dict[str, Any]:
    """Get candidate pipeline status for a requisition.

    Returns current pipeline status showing candidate distribution by source.

    ERROR BEHAVIOR: 20% chance of returning a 404-style error dict.
    Tests retry logic and error recovery.
    """
    if _rng.random() < 0.2:
        return {
            "status_code": 404,
            "error": True,
            "message": "Resource temporarily unavailable",
        }

    err = _check_requisition(requisition_id)
    if err:
        return err

    loader = get_data_loader()
    data = loader.get_similar_requisitions(requisition_id)
    volume_by_source = data.groupby("source_name").size().to_dict()

    return {
        "requisition_id": requisition_id,
        "pipeline": {k: int(v) for k, v in volume_by_source.items()},
        "total_candidates": int(len(data)),
    }


# ── Test 31: HTTP 500 with valid body ────────────────────────────────────────

def get_source_sla_check(requisition_id: str) -> Dict[str, Any]:
    """Run a quick SLA status check across all sourcing channels.

    Returns SLA metrics per source for rapid status assessment.

    ERROR BEHAVIOR: Returns HTTP 500-style error dict but includes valid
    data in the body. Tests agent ability to use response body despite
    error status code.
    """
    err = _check_requisition(requisition_id)
    if err:
        return err

    loader = get_data_loader()
    data = loader.get_similar_requisitions(requisition_id)
    reviewed = data[data["reviewed"]]
    metrics = []
    for source, group in reviewed.groupby("source_name"):
        sla_pct = int(round(group["sla_met"].mean() * 100))
        metrics.append({"source_name": source, "sla_percentage": sla_pct})

    return {
        "status_code": 500,
        "error": True,
        "message": "Internal server error",
        "body": {"metrics": metrics},
    }


# ── Test 32: HTTP 503 Service Unavailable ────────────────────────────────────

def get_funnel_status(requisition_id: str) -> Dict[str, Any]:
    """Get the current funnel status for a requisition.

    Returns real-time funnel pipeline status showing conversion at each stage.

    ERROR BEHAVIOR: Always returns 503 with retry-after info.
    Tests service unavailable handling.
    """
    return {
        "status_code": 503,
        "error": True,
        "message": "Service temporarily unavailable. The funnel analytics engine is undergoing maintenance.",
        "retry_after_seconds": 300,
        "expected_recovery": "2025-05-01T12:00:00Z",
    }


# ── Test 33: HTTP 429 Rate Limited ──────────────────────────────────────────

def get_bulk_source_data(requisition_id: str) -> Dict[str, Any]:
    """Pull bulk source data for all requisitions.

    Returns comprehensive source data across all requisitions in the system.

    ERROR BEHAVIOR: Returns 429 after 3rd call (tracked via module-level counter).
    Tests rate limit handling.
    """
    key = "get_bulk_source_data"
    _call_counts[key] = _call_counts.get(key, 0) + 1

    if _call_counts[key] > 3:
        return {
            "status_code": 429,
            "error": True,
            "message": "Rate limit exceeded. Maximum 3 calls per session.",
            "retry_after_seconds": 60,
            "limit": 3,
            "remaining": 0,
        }

    err = _check_requisition(requisition_id)
    if err:
        return err

    loader = get_data_loader()
    data = loader.get_similar_requisitions(requisition_id)
    summary = {}
    for source, group in data.groupby("source_name"):
        summary[source] = {
            "total_candidates": int(len(group)),
            "total_hires": int(group["hired"].sum()),
            "reviewed": int(group["reviewed"].sum()),
        }

    return {
        "requisition_id": requisition_id,
        "sources": summary,
        "call_number": _call_counts[key],
    }


# ── Test 36: Missing required fields ────────────────────────────────────────

def get_source_metrics_lite(requisition_id: str) -> Dict[str, Any]:
    """Get a lightweight summary of source metrics.

    Returns a compact view of per-source metrics for quick analysis.

    ERROR BEHAVIOR: Response missing `source_name` field in metrics entries.
    Tests agent handling of incomplete/partial data.
    """
    err = _check_requisition(requisition_id)
    if err:
        return err

    loader = get_data_loader()
    data = loader.get_similar_requisitions(requisition_id)
    metrics = []
    for source, group in data.groupby("source_name"):
        # Intentionally omit source_name
        metrics.append({
            "candidate_count": int(len(group)),
            "hire_count": int(group["hired"].sum()),
            "sla_met_count": int(group[group["reviewed"]]["sla_met"].sum()),
        })

    return {
        "requisition_id": requisition_id,
        "metrics": metrics,
        "note": "Lightweight view — some fields may be omitted for performance.",
    }


# ── Test 37: Wrong field types in response ──────────────────────────────────

def get_volume_report(requisition_id: str) -> Dict[str, Any]:
    """Generate a volume report for a requisition.

    Returns candidate volume statistics broken down by source.

    ERROR BEHAVIOR: `candidate_count` returned as string instead of int.
    Tests type coercion handling.
    """
    err = _check_requisition(requisition_id)
    if err:
        return err

    loader = get_data_loader()
    data = loader.get_similar_requisitions(requisition_id)
    metrics = []
    for source, group in data.groupby("source_name"):
        metrics.append({
            "source_name": source,
            "candidate_count": str(len(group)),  # String instead of int
            "hire_count": str(int(group["hired"].sum())),  # String instead of int
            "review_rate": f"{group['reviewed'].mean() * 100:.1f}%",
        })

    return {
        "requisition_id": requisition_id,
        "metrics": metrics,
        "total_candidates": str(len(data)),  # String instead of int
    }


# ── Test 38: Large response (1000 records) ──────────────────────────────────

def get_full_candidate_details(requisition_id: str) -> Dict[str, Any]:
    """Get full candidate details for a requisition.

    Returns comprehensive candidate-level data for detailed analysis.

    ERROR BEHAVIOR: Response contains 1000 pre-generated candidate records.
    Tests agent handling of large payloads.
    """
    err = _check_requisition(requisition_id)
    if err:
        return err

    # Load pre-generated fixture
    fixture_paths = [
        Path(__file__).parent.parent.parent / "data" / "large_response_fixture.json",
        Path("./data/large_response_fixture.json"),
    ]

    for path in fixture_paths:
        if path.exists():
            with open(path, "r") as f:
                records = json.load(f)
            return {
                "requisition_id": requisition_id,
                "total_records": len(records),
                "candidates": records,
            }

    # Fallback: generate minimal records if fixture missing
    return {
        "requisition_id": requisition_id,
        "total_records": 0,
        "candidates": [],
        "warning": "Large response fixture not found",
    }


# ── Test 39: Unicode and special characters ─────────────────────────────────

def get_source_directory(requisition_id: str) -> Dict[str, Any]:
    """Show the source directory for a requisition.

    Returns a directory listing of all sourcing channels with their metadata.

    ERROR BEHAVIOR: Source names contain emoji, CJK characters, Arabic text.
    Tests unicode handling.
    """
    err = _check_requisition(requisition_id)
    if err:
        return err

    return {
        "requisition_id": requisition_id,
        "sources": [
            {"name": "LinkedIn \U0001F4BC", "region": "Global", "status": "active"},
            {"name": "Dice \U0001F3B2", "region": "North America", "status": "active"},
            {"name": "\u62db\u8058\u7f51 (Zhaopin)", "region": "\u4e2d\u56fd", "status": "active"},
            {"name": "\u0628\u064a\u062a.\u0643\u0648\u0645 (Bayt)", "region": "\u0627\u0644\u0634\u0631\u0642 \u0627\u0644\u0623\u0648\u0633\u0637", "status": "active"},
            {"name": "GitHub \U0001F431\u200D\U0001F4BB", "region": "Global", "status": "active"},
            {"name": "R\u00e9f\u00e9rence\u2122", "region": "Europe", "status": "inactive"},
            {"name": "\u2605 Top Talent \u2605", "region": "APAC", "status": "active"},
        ],
        "total_sources": 7,
    }


# ── Test 41: Extra undocumented fields (20 extra fields) ─────────────────────

def get_sla_extended(requisition_id: str, source_name: str = "Dice") -> Dict[str, Any]:
    """Get extended SLA data for a specific sourcing channel.

    Returns SLA metrics with additional analytics for the given source.

    ERROR BEHAVIOR: Response includes 20 undocumented extra fields
    beyond what the schema describes. Tests agent ability to ignore
    noise and extract relevant data.
    """
    err = _check_requisition(requisition_id)
    if err:
        return err

    loader = get_data_loader()
    data = loader.get_similar_requisitions(requisition_id)
    source_data = data[(data["reviewed"]) & (data["source_name"] == source_name)]

    sla_pct = int(round(source_data["sla_met"].mean() * 100)) if len(source_data) > 0 else 0

    return {
        "requisition_id": requisition_id,
        "source_name": source_name,
        "sla_percentage": sla_pct,
        # Undocumented extra fields
        "_internal_id": "sla-ext-7f3a9b2c",
        "_cache_ttl": 3600,
        "_version": "2.3.1",
        "_debug_query_ms": 42,
        "_shard_id": 3,
        "_region": "us-east-1",
        "_feature_flags": ["sla_v2", "extended_metrics"],
        "_experiment_group": "control",
        "_sampling_rate": 0.95,
        "_data_quality_score": 0.98,
        "_last_recomputed": "2025-04-29T03:00:00Z",
        "_computation_engine": "spark-3.5",
        "_model_version": "sla-impact-v1.4.2",
        "_confidence_interval": [sla_pct - 3, sla_pct + 3],
        "_p_value": 0.023,
        "_sample_size": int(len(source_data)),
        "_outliers_removed": 2,
        "_normalization_method": "min-max",
        "_correlation_with_hires": 0.67,
        "_seasonal_adjustment": True,
    }


# ── Test 43: Undocumented error format ──────────────────────────────────────

def get_requisition_details(requisition_id: str) -> Dict[str, Any]:
    """Get detailed information for a specific requisition.

    Returns comprehensive requisition metadata and status.

    ERROR BEHAVIOR: Returns non-standard error format `{"err": "not_found"}`
    instead of the standard `RequisitionNotFoundResponse`.
    Tests non-standard error parsing.
    """
    loader = get_data_loader()
    if not loader.is_valid_requisition(requisition_id):
        return {"err": "not_found", "req": requisition_id}

    data = loader.get_by_requisition(requisition_id)
    row = data.iloc[0]

    return {
        "requisition_id": requisition_id,
        "department": str(row.get("department", "Unknown")),
        "seniority_level": str(row.get("seniority_level", "Unknown")),
        "total_candidates": int(len(data)),
        "sources_used": list(data["source_name"].unique()),
    }


# ── Test 44: Undocumented pagination ─────────────────────────────────────────

def list_all_sources(requisition_id: str) -> Dict[str, Any]:
    """List all available sourcing channels.

    Returns a paginated list of all sourcing channels in the system.

    ERROR BEHAVIOR: Response includes `next_page` token not described
    in any schema. Tests pagination detection and handling.
    """
    err = _check_requisition(requisition_id)
    if err:
        return err

    loader = get_data_loader()
    data = loader.get_similar_requisitions(requisition_id)
    sources = sorted(data["source_name"].unique())

    # Return first 3 with pagination token
    page_size = 3
    page = sources[:page_size]

    result: Dict[str, Any] = {
        "requisition_id": requisition_id,
        "sources": [{"name": s, "index": i} for i, s in enumerate(page)],
        "total_count": len(sources),
        "page_size": page_size,
        "page": 1,
    }

    if len(sources) > page_size:
        result["next_page"] = "eyJvZmZzZXQiOjMsInJlcV9pZCI6IjA1OTU4QlIifQ=="
        result["has_more"] = True
    else:
        result["has_more"] = False

    return result


# ── Test 45: Undocumented rate limiting headers ──────────────────────────────

def get_batch_metrics(requisition_id: str) -> Dict[str, Any]:
    """Fetch batch metrics for all sourcing channels.

    Returns aggregated metrics across all sources with rate limit information.

    ERROR BEHAVIOR: Response includes X-RateLimit style headers embedded
    in the response body. Tests rate limit awareness.
    """
    err = _check_requisition(requisition_id)
    if err:
        return err

    loader = get_data_loader()
    data = loader.get_similar_requisitions(requisition_id)

    metrics = {}
    for source, group in data.groupby("source_name"):
        metrics[source] = {
            "candidates": int(len(group)),
            "hires": int(group["hired"].sum()),
            "reviewed": int(group["reviewed"].sum()),
        }

    return {
        "requisition_id": requisition_id,
        "metrics": metrics,
        # Rate limit info embedded in response body
        "X-RateLimit-Limit": 100,
        "X-RateLimit-Remaining": 97,
        "X-RateLimit-Reset": "2025-05-01T00:00:00Z",
        "X-RateLimit-Window": "1h",
    }
