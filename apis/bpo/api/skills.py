"""Skills APIs - compute skill-related metrics from actual data."""

from typing import Dict, List, Any, Optional, Union
import pandas as pd
from loguru import logger
from .data_loader import get_data_loader
from .schemas import (
    RequisitionNotFoundResponse,
    SkillAnalysisResponse,
    SkillImpactFillRateResponse,
    SkillImpactSLAResponse,
    SkillRelevanceResponse,
    SuccessfulPostingResponse,
    DataSourcesResponse,
    SkillJustificationData,
    SkillJustificationImpact,
    SuccessCriteria,
)
try:
    from ..config import BPO_LOG_API_CALLS
except ImportError:
    BPO_LOG_API_CALLS = False


def _log_api_call(msg: str) -> None:
    """Log API call if BPO_LOG_API_CALLS is enabled."""
    if BPO_LOG_API_CALLS:
        logger.info(msg)


def _check_requisition_valid(requisition_id: str) -> Optional[RequisitionNotFoundResponse]:
    """
    Check if a requisition ID is valid. Returns None if valid,
    or an error response model if invalid.
    """
    loader = get_data_loader()
    if not loader.is_valid_requisition(requisition_id):
        suggestions = loader.get_suggested_requisitions(requisition_id)
        return RequisitionNotFoundResponse(
            error="requisition_not_found",
            message=f"No job can be found with the ID {requisition_id}.",
            suggested_requisition_ids=suggestions,
        )
    return None


def get_skill_analysis(requisition_id: str) -> Union[SkillAnalysisResponse, RequisitionNotFoundResponse]:
    """
    Provides statistical indicators for each skill associated with the requisition,
    enabling an LLM or analyst to decide whether a skill should be retained,
    removed, or reconsidered.

    Args:
        requisition_id: The job requisition ID.

    Returns:
        Dict with historical counts and SLA correlation per skill.
    """
    _log_api_call(f"API call: get_skill_analysis(requisition_id={requisition_id})")

    # Check if requisition ID is valid
    error = _check_requisition_valid(requisition_id)
    if error:
        return error

    loader = get_data_loader()
    data = loader.get_similar_requisitions(requisition_id)

    # Get all unique skills across all candidates
    all_skills = []
    for skills_list in data['skills_parsed']:
        all_skills.extend(skills_list)

    skill_counts = pd.Series(all_skills).value_counts()

    # For each skill, compute SLA correlation
    historical_skills = []
    for skill, count in skill_counts.head(10).items():  # Top 10 skills
        # Filter to reviewed candidates only (SLA only applies to reviewed candidates)
        reviewed_data = data[data['reviewed']]

        # Get candidates with and without this skill
        has_skill = reviewed_data[reviewed_data['skills_parsed'].apply(lambda x: skill in x)]
        no_skill = reviewed_data[reviewed_data['skills_parsed'].apply(lambda x: skill not in x)]

        # Calculate SLA rates
        sla_with = has_skill['sla_met'].mean() if len(has_skill) > 0 else 0
        sla_without = no_skill['sla_met'].mean() if len(no_skill) > 0 else 0

        # Determine correlation
        diff = sla_with - sla_without
        if diff < -0.10:
            correlation = "highly negative impact on SLA"
        elif diff < 0:
            correlation = "slightly negative impact on SLA"
        elif diff > 0.10:
            correlation = "highly positive impact on SLA"
        elif diff > 0:
            correlation = "slightly positive impact on SLA"
        else:
            correlation = "no impact on SLA"

        historical_skills.append({
            "name": skill,
            "skill_occurrence": int(count),
            "correlation": correlation
        })

    num_jobs = data['requisition_id'].nunique()

    return SkillAnalysisResponse(
        historical_jobs=num_jobs,
        input_skills=[],  # Would come from requisition details
        historical_skills_with_analysis=historical_skills,
    )


def get_skill_impact_fill_rate(requisition_id: str, skill_name: str) -> Union[SkillImpactFillRateResponse, RequisitionNotFoundResponse]:
    """
    Evaluates how the inclusion of a specific skill affects requisition
    fill-rate metrics and candidate pool size.

    Args:
        requisition_id: The job requisition ID.
        skill_name: The skill to evaluate.

    Returns:
        Impact metrics with and without the skill.
    """
    _log_api_call(f"API call: get_skill_impact_fill_rate(requisition_id={requisition_id}, skill_name={skill_name})")

    # Check if requisition ID is valid
    error = _check_requisition_valid(requisition_id)
    if error:
        return error

    loader = get_data_loader()
    data = loader.get_similar_requisitions(requisition_id)

    # Split data by whether requisitions included this skill
    has_skill_reqs = data[data['skills_parsed'].apply(lambda x: skill_name in x)]['requisition_id'].unique()
    no_skill_reqs = data[~data['requisition_id'].isin(has_skill_reqs)]['requisition_id'].unique()

    def calc_metrics(req_ids):
        if len(req_ids) == 0:
            return {"fill_rate_percentage": 0, "time_to_fill_days": 0, "candidate_pool_size": 0}

        req_data = data[data['requisition_id'].isin(req_ids)]

        # Fill rate: % of reqs that got at least one hire
        reqs_with_hires = req_data[req_data['hired']]['requisition_id'].nunique()
        fill_rate = reqs_with_hires / len(req_ids) * 100

        # Time to fill: average days from applied to hired
        hired = req_data[req_data['hired']]
        if len(hired) > 0:
            time_to_fill = (hired['hire_date'] - hired['applied_at']).dt.days.mean()
        else:
            time_to_fill = 0

        # Candidate pool size
        pool_size = len(req_data)

        return {
            "fill_rate_percentage": round(fill_rate, 1),
            "time_to_fill_days": int(time_to_fill),
            "candidate_pool_size": pool_size
        }

    with_skill = calc_metrics(has_skill_reqs)
    without_skill = calc_metrics(no_skill_reqs)

    return SkillImpactFillRateResponse(
        skill_name=skill_name,
        impact=with_skill,
        compared_to_baseline=without_skill,
    )


def get_skill_impact_sla(requisition_id: str, skill_name: str) -> Union[SkillImpactSLAResponse, RequisitionNotFoundResponse]:
    """
    Analyzes how a skill affects SLA achievement rate.

    Args:
        requisition_id: The job requisition ID.
        skill_name: The skill being analyzed.

    Returns:
        Success percentages with/without the skill and the delta.
    """
    _log_api_call(f"API call: get_skill_impact_sla(requisition_id={requisition_id}, skill_name={skill_name})")

    # Check if requisition ID is valid
    error = _check_requisition_valid(requisition_id)
    if error:
        return error

    loader = get_data_loader()
    data = loader.get_similar_requisitions(requisition_id)

    # Filter to reviewed candidates only (SLA only applies to reviewed candidates)
    reviewed_data = data[data['reviewed']]

    # Get candidates with and without this skill
    has_skill = reviewed_data[reviewed_data['skills_parsed'].apply(lambda x: skill_name in x)]
    no_skill = reviewed_data[reviewed_data['skills_parsed'].apply(lambda x: skill_name not in x)]

    sla_with = round(has_skill['sla_met'].mean() * 100) if len(has_skill) > 0 else 0
    sla_without = round(no_skill['sla_met'].mean() * 100) if len(no_skill) > 0 else 0

    return SkillImpactSLAResponse(
        requisition_id=requisition_id,
        skill_name=skill_name,
        sla_achievement_with_skill=sla_with,
        sla_achievement_without_skill=sla_without,
        delta=sla_with - sla_without,
    )


def get_skill_relevance_justification(requisition_id: str, skill_name: str) -> Union[SkillRelevanceResponse, RequisitionNotFoundResponse]:
    """
    Explains whether a skill is relevant and why, based on historical hiring
    success and outcome data.

    Args:
        requisition_id: The job requisition ID.
        skill_name: The skill being justified.

    Returns:
        Relevance determination with justification.
    """
    _log_api_call(f"API call: get_skill_relevance_justification(requisition_id={requisition_id}, skill_name={skill_name})")

    # Check if requisition ID is valid
    error = _check_requisition_valid(requisition_id)
    if error:
        return error

    # Get both SLA and fill rate impacts
    sla_impact = get_skill_impact_sla(requisition_id, skill_name)
    fill_impact = get_skill_impact_fill_rate(requisition_id, skill_name)

    # Determine relevance based on both metrics
    is_relevant = False
    if sla_impact.delta > 5 or fill_impact.impact.fill_rate_percentage > fill_impact.compared_to_baseline.fill_rate_percentage * 1.2:
        is_relevant = True

    justification = SkillJustificationData(
        requisition_id=requisition_id,
        skill_name=skill_name,
        sla_achievement_with_skill=sla_impact.sla_achievement_with_skill,
        sla_achievement_without_skill=sla_impact.sla_achievement_without_skill,
        delta=sla_impact.delta,
        impact=SkillJustificationImpact(
            fill_rate_percentage=fill_impact.impact.fill_rate_percentage,
            time_to_fill_days=fill_impact.impact.time_to_fill_days,
            candidate_pool_size=fill_impact.impact.candidate_pool_size,
        ),
        compared_to_baseline=SkillJustificationImpact(
            fill_rate_percentage=fill_impact.compared_to_baseline.fill_rate_percentage,
            time_to_fill_days=fill_impact.compared_to_baseline.time_to_fill_days,
            candidate_pool_size=fill_impact.compared_to_baseline.candidate_pool_size,
        ),
    )

    return SkillRelevanceResponse(
        requisition_id=requisition_id,
        skill_name=skill_name,
        is_relevant=is_relevant,
        justification=justification,
    )


def get_successful_posting_criteria() -> SuccessfulPostingResponse:
    """
    Returns the business definition of a successful job posting,
    including thresholds and benchmarks for success.

    Returns:
        Success criteria thresholds.
    """
    _log_api_call("API call: get_successful_posting_criteria()")

    return SuccessfulPostingResponse(
        criteria=SuccessCriteria(
            time_to_fill_threshold_days=90,
            offer_acceptance_rate_min=50,
            sla_compliance_min=80,
            candidate_quality_rating_avg=3.5,
        ),
        justification="Based on historical performance benchmarks and industry standards",
    )


def get_data_sources_used(requisition_id: str) -> Union[DataSourcesResponse, RequisitionNotFoundResponse]:
    """
    Lists the datasets and ML models used to make hiring recommendations
    for a requisition.

    Args:
        requisition_id: The job requisition ID.

    Returns:
        Data sources and models used.
    """
    _log_api_call(f"API call: get_data_sources_used(requisition_id={requisition_id})")

    # Check if requisition ID is valid
    error = _check_requisition_valid(requisition_id)
    if error:
        return error

    return DataSourcesResponse(
        requisition_id=requisition_id,
        datasets_used=[
            "Historical hiring success data",
            "Requisition skill tagging",
            "Funnel conversion metrics",
            "Candidate quality feedback",
        ],
        models_involved=[
            "Skill relevance classifier",
            "SLA impact regression model",
            "Funnel conversion recommender",
        ],
    )
