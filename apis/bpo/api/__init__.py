"""API implementations for BPO tools."""

from .candidate_source import (
    get_sla_per_source,
    get_total_hires_by_source,
    get_candidate_volume_by_source,
    get_funnel_conversion_by_source,
    get_metadata_and_timeframe,
    get_definitions_and_methodology,
    get_source_recommendation_summary,
)

from .skills import (
    get_skill_analysis,
    get_skill_impact_fill_rate,
    get_skill_impact_sla,
    get_skill_relevance_justification,
    get_successful_posting_criteria,
    get_data_sources_used,
)

__all__ = [
    "get_sla_per_source",
    "get_total_hires_by_source",
    "get_candidate_volume_by_source",
    "get_funnel_conversion_by_source",
    "get_metadata_and_timeframe",
    "get_definitions_and_methodology",
    "get_source_recommendation_summary",
    "get_skill_analysis",
    "get_skill_impact_fill_rate",
    "get_skill_impact_sla",
    "get_skill_relevance_justification",
    "get_successful_posting_criteria",
    "get_data_sources_used",
]
