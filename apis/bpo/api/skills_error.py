"""Error-prone skills API variants for testing agent resilience.

Each function has a unique, plausible intent and embeds a specific error behavior.
Completely independent from original APIs — accesses DataLoader directly.
"""

import json
import random
from typing import Any, Dict, Optional

from .data_loader import get_data_loader

# Seeded RNG for reproducible probabilistic behavior
_rng = random.Random(42)


def _check_requisition(requisition_id: str) -> Optional[Dict[str, Any]]:
    """Return error dict if requisition invalid, else None."""
    loader = get_data_loader()
    if not loader.is_valid_requisition(requisition_id):
        return {
            "error": "requisition_not_found",
            "message": f"Requisition {requisition_id} not found",
        }
    return None


# ── Test 27: Type mismatch — string instead of structured list ──────────────

def get_skill_summary(requisition_id: str) -> str:
    """Get a quick text summary of skills needed for a requisition.

    Returns a concise comma-separated skill overview.

    ERROR BEHAVIOR: Returns a plain comma-separated string instead of
    structured SkillAnalysisResponse. Tests type mismatch handling.
    """
    err = _check_requisition(requisition_id)
    if err:
        return json.dumps(err)

    loader = get_data_loader()
    data = loader.get_similar_requisitions(requisition_id)
    all_skills: set = set()
    for skills_list in data["skills_parsed"].dropna():
        if isinstance(skills_list, list):
            all_skills.update(skills_list)

    return ", ".join(sorted(all_skills))


# ── Test 34: Missing output schema — untyped dict ───────────────────────────

def get_model_registry(requisition_id: str) -> Dict[str, Any]:
    """Check which ML models are registered for a given requisition.

    Returns model registry information including versions and status.

    ERROR BEHAVIOR: No Pydantic output schema — returns a plain dict
    with dynamically typed fields. Tests schema inference.
    """
    err = _check_requisition(requisition_id)
    if err:
        return err

    return {
        "requisition_id": requisition_id,
        "models": [
            {
                "name": "Skill relevance classifier",
                "version": "2.1.0",
                "status": "active",
                "last_trained": "2024-11-15",
                "accuracy": 0.87,
            },
            {
                "name": "SLA impact regression model",
                "version": "1.4.2",
                "status": "active",
                "last_trained": "2024-10-01",
                "r_squared": 0.72,
            },
            {
                "name": "Funnel conversion recommender",
                "version": "3.0.0-beta",
                "status": "staging",
                "last_trained": "2025-01-20",
                "precision": 0.81,
            },
        ],
        "registry_updated": "2025-04-29",
    }


# ── Test 35: Missing input schema — undocumented params ─────────────────────

def get_skill_lookup(requisition_id: str, skill_name: str = None,
                     include_history: bool = False,
                     format: str = "json") -> Dict[str, Any]:
    """Look up a specific skill and its metrics for a requisition.

    ERROR BEHAVIOR: Accepts undocumented parameters (include_history, format)
    not described in the tool schema. Tests agent handling of extra params.
    """
    err = _check_requisition(requisition_id)
    if err:
        return err

    loader = get_data_loader()
    data = loader.get_similar_requisitions(requisition_id)

    # Find skill occurrence
    total = 0
    for skills_list in data["skills_parsed"].dropna():
        if isinstance(skills_list, list) and skill_name in skills_list:
            total += 1

    result = {
        "requisition_id": requisition_id,
        "skill_name": skill_name,
        "occurrence_count": total,
        "total_candidates": len(data),
        "occurrence_rate": round(total / len(data) * 100, 1) if len(data) > 0 else 0,
    }

    if include_history:
        result["history"] = {
            "first_seen": "2023-10-09",
            "trend": "stable",
            "quarterly_counts": [total // 4] * 4,
        }

    return result


# ── Test 40: Deeply nested JSON (15 levels) ─────────────────────────────────

def get_skill_deep_analysis(requisition_id: str) -> Dict[str, Any]:
    """Get a deep analysis breakdown of skills with detailed sub-categories.

    Returns comprehensive multi-level skill categorization and metrics.

    ERROR BEHAVIOR: Response is nested 15 levels deep.
    Tests agent ability to navigate deeply nested structures.
    """
    err = _check_requisition(requisition_id)
    if err:
        return err

    loader = get_data_loader()
    data = loader.get_similar_requisitions(requisition_id)

    # Collect top skills
    all_skills: list = []
    for skills_list in data["skills_parsed"].dropna():
        if isinstance(skills_list, list):
            all_skills.extend(skills_list)

    from collections import Counter
    skill_counts = Counter(all_skills)
    top_skills = skill_counts.most_common(5)

    # Build deeply nested structure (15 levels)
    def nest(depth: int, skill_name: str, count: int) -> Dict[str, Any]:
        if depth <= 0:
            return {"skill": skill_name, "count": count}
        return {
            "level": depth,
            "metadata": {"type": f"analysis_layer_{depth}"},
            "data": nest(depth - 1, skill_name, count),
        }

    skills_nested = [
        nest(15, name, count) for name, count in top_skills
    ]

    return {
        "requisition_id": requisition_id,
        "analysis_version": "3.0",
        "results": {
            "nested_skills": skills_nested,
            "total_depth": 15,
        },
    }


# ── Test 42: Input schema mismatch — expects skill_id but docs say skill_name

def analyze_skill_match(requisition_id: str, skill_id: str) -> Dict[str, Any]:
    """Check if a skill is a good match for a requisition.

    Args:
        requisition_id: The job requisition ID.
        skill_id: The skill identifier to check.

    ERROR BEHAVIOR: Function signature says `skill_id` but tool description
    and documentation say `skill_name`. Tests agent adaptation to mismatched
    parameter names.
    """
    err = _check_requisition(requisition_id)
    if err:
        return err

    # Treat skill_id as skill_name (the mismatch)
    skill_name = skill_id

    loader = get_data_loader()
    data = loader.get_similar_requisitions(requisition_id)
    reviewed = data[data["reviewed"]]

    has_skill = reviewed[reviewed["skills_parsed"].apply(lambda x: skill_name in x)]
    no_skill = reviewed[reviewed["skills_parsed"].apply(lambda x: skill_name not in x)]

    sla_with = round(has_skill["sla_met"].mean() * 100) if len(has_skill) > 0 else 0
    sla_without = round(no_skill["sla_met"].mean() * 100) if len(no_skill) > 0 else 0

    total_with_skill = sum(
        1 for sl in data["skills_parsed"].dropna()
        if isinstance(sl, list) and skill_name in sl
    )

    match_score = min(100, int(
        (total_with_skill / len(data) * 50 if len(data) > 0 else 0)
        + (max(0, sla_with - sla_without))
    ))

    return {
        "requisition_id": requisition_id,
        "skill_id": skill_name,
        "match_score": match_score,
        "sla_delta": sla_with - sla_without,
        "occurrence_rate": round(total_with_skill / len(data) * 100, 1) if len(data) > 0 else 0,
        "recommendation": "good match" if match_score >= 50 else "weak match",
    }
