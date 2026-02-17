"""Pydantic schemas for BPO API responses."""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any


# ============================================================================
# Error Response Models
# ============================================================================

class RequisitionNotFoundResponse(BaseModel):
    """Response returned when a requisition ID is not found."""
    error: str
    message: str
    suggested_requisition_ids: List[str]


# ============================================================================
# Candidate Source Response Models
# ============================================================================

class SLAMetric(BaseModel):
    """SLA metric for a single source."""
    source_name: str
    sla_percentage: int


class SLAPerSourceResponse(BaseModel):
    """Response for get_sla_per_source API."""
    metrics: List[SLAMetric]


class HireMetric(BaseModel):
    """Hire metric for a single source."""
    source_name: str
    total_hires: int


class TotalHiresBySourceResponse(BaseModel):
    """Response for get_total_hires_by_source API."""
    job_id: str
    metrics: List[HireMetric]
    total_hires: int


class VolumeMetric(BaseModel):
    """Volume metric for a single source."""
    source_name: str
    candidate_volume: int
    percentage: int


class CandidateVolumeResponse(BaseModel):
    """Response for get_candidate_volume_by_source API."""
    job_id: str
    total_candidate_volume: int
    metrics: List[VolumeMetric]
    heading: str


class FunnelMetric(BaseModel):
    """Funnel conversion metric for a single source."""
    source_name: str
    first_round_review_percentage: float
    interview_rate: float
    offer_acceptance_rate: float


class FunnelConversionResponse(BaseModel):
    """Response for get_funnel_conversion_by_source API."""
    job_id: str
    metrics: List[FunnelMetric]


class MetadataResponse(BaseModel):
    """Response for get_metadata_and_timeframe API."""
    job_id: str
    time_frame_start: str
    time_frame_end: str
    data_last_updated: str
    total_requisitions_analysed: int


class DefinitionsResponse(BaseModel):
    """Response for get_definitions_and_methodology API."""
    job_id: str
    definitions: Dict[str, str]
    calculation_notes: str
    top_metrics_considered: List[str]


class SourceSummaryMetric(BaseModel):
    """Summary metric for a single source."""
    source_name: str
    jobs_filled_percentage: int
    first_round_review_percentage: int
    offer_acceptance_rate: int
    total_hires: int


class SourceRecommendationResponse(BaseModel):
    """Response for get_source_recommendation_summary API."""
    total_requisitions: int
    metrics: List[SourceSummaryMetric]


# ============================================================================
# Skills Response Models
# ============================================================================

class SkillWithAnalysis(BaseModel):
    """Skill with historical analysis."""
    name: str
    skill_occurrence: int
    correlation: str


class SkillAnalysisResponse(BaseModel):
    """Response for get_skill_analysis API."""
    historical_jobs: int
    input_skills: List[Any]
    historical_skills_with_analysis: List[SkillWithAnalysis]


class ImpactMetrics(BaseModel):
    """Impact metrics for skill analysis."""
    fill_rate_percentage: float
    time_to_fill_days: int
    candidate_pool_size: int


class SkillImpactFillRateResponse(BaseModel):
    """Response for get_skill_impact_fill_rate API."""
    skill_name: str
    impact: ImpactMetrics
    compared_to_baseline: ImpactMetrics


class SkillImpactSLAResponse(BaseModel):
    """Response for get_skill_impact_sla API."""
    requisition_id: str
    skill_name: str
    sla_achievement_with_skill: int
    sla_achievement_without_skill: int
    delta: int


class SkillJustificationImpact(BaseModel):
    """Impact metrics within justification."""
    fill_rate_percentage: float
    time_to_fill_days: int
    candidate_pool_size: int


class SkillJustificationData(BaseModel):
    """Justification data for skill relevance."""
    requisition_id: str
    skill_name: str
    sla_achievement_with_skill: int
    sla_achievement_without_skill: int
    delta: int
    impact: SkillJustificationImpact
    compared_to_baseline: SkillJustificationImpact


class SkillRelevanceResponse(BaseModel):
    """Response for get_skill_relevance_justification API."""
    requisition_id: str
    skill_name: str
    is_relevant: bool
    justification: SkillJustificationData


class SuccessCriteria(BaseModel):
    """Success criteria thresholds."""
    time_to_fill_threshold_days: int
    offer_acceptance_rate_min: int
    sla_compliance_min: int
    candidate_quality_rating_avg: float


class SuccessfulPostingResponse(BaseModel):
    """Response for get_successful_posting_criteria API."""
    criteria: SuccessCriteria
    justification: str


class DataSourcesResponse(BaseModel):
    """Response for get_data_sources_used API."""
    requisition_id: str
    datasets_used: List[str]
    models_involved: List[str]
