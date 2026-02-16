"""FastMCP server exposing BPO APIs as tools."""

import json
import os
import sys
from pathlib import Path
from typing import Optional, List, Union, Any, Dict
import fastmcp
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from loguru import logger

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from apis.bpo.api.schemas import (
    RequisitionNotFoundResponse,
    SLAPerSourceResponse,
    TotalHiresBySourceResponse,
    CandidateVolumeResponse,
    FunnelConversionResponse,
    MetadataResponse,
    DefinitionsResponse,
    SourceRecommendationResponse,
    SkillAnalysisResponse,
    SkillImpactFillRateResponse,
    SkillImpactSLAResponse,
    SkillRelevanceResponse,
    SuccessfulPostingResponse,
    DataSourcesResponse,
)

# Tool call tracking file - allows evaluation to see which tools were called
TOOL_CALLS_FILE = Path("/tmp/bpo_mcp_tool_calls.json")


def clear_tool_calls() -> None:
    """Clear the tool calls tracking file."""
    try:
        with open(TOOL_CALLS_FILE, 'w') as f:
            json.dump([], f)
        logger.info("Cleared tool calls file")
    except Exception as e:
        logger.error(f"Failed to clear tool calls file: {e}")


def get_tool_calls() -> list:
    """Get the list of tracked tool calls."""
    try:
        if TOOL_CALLS_FILE.exists():
            with open(TOOL_CALLS_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to read tool calls: {e}")
    return []


def track_tool_call(tool_name: str) -> None:
    """Record a tool call to the tracking file."""
    try:
        # Read existing calls
        calls = []
        if TOOL_CALLS_FILE.exists():
            with open(TOOL_CALLS_FILE, 'r') as f:
                calls = json.load(f)

        # Add new call
        calls.append(tool_name)

        # Write back
        with open(TOOL_CALLS_FILE, 'w') as f:
            json.dump(calls, f)

        logger.info(f"Tracked tool call: {tool_name}")
    except Exception as e:
        logger.error(f"Failed to track tool call: {e}")

# Import all API functions
from apis.bpo.api.candidate_source import (
    get_sla_per_source,
    get_total_hires_by_source,
    get_candidate_volume_by_source,
    get_funnel_conversion_by_source,
    get_metadata_and_timeframe,
    get_definitions_and_methodology,
    get_source_recommendation_summary,
)

from apis.bpo.api.skills import (
    get_skill_analysis,
    get_skill_impact_fill_rate,
    get_skill_impact_sla,
    get_skill_relevance_justification,
    get_successful_posting_criteria,
    get_data_sources_used,
)

# Import error-prone API variants
from apis.bpo.api import skills_error, candidate_source_error

# Create MCP server
# Note: Server name should match the key in evaluation/mcp_servers.yaml
mcp = fastmcp.FastMCP("bpo_benchmark")


# Candidate Source Tools
@mcp.tool()
def candidate_source_sla_per_source(requisition_id: str) -> Union[SLAPerSourceResponse, RequisitionNotFoundResponse]:
    """
    Retrieves the SLA percentage for each sourcing channel.

    Args:
        requisition_id: The specific requisition ID to filter SLA data for.

    Returns:
        A dictionary with source names and their SLA percentages.
    """
    track_tool_call("candidate_source_sla_per_source")
    logger.info(f"MCP tool called: candidate_source_sla_per_source(requisition_id={requisition_id})")
    return get_sla_per_source(requisition_id)


@mcp.tool()
def candidate_source_total_hires_by_source(requisition_id: str) -> Union[TotalHiresBySourceResponse, RequisitionNotFoundResponse]:
    """
    Retrieves the total number of hires per sourcing channel.

    Args:
        requisition_id: The specific requisition ID to filter hiring data for.

    Returns:
        A dictionary with source names and total hires.
    """
    track_tool_call("candidate_source_total_hires_by_source")
    return get_total_hires_by_source(requisition_id)


@mcp.tool()
def candidate_source_candidate_volume_by_source(
    requisition_id: str,
    sources: Optional[List[str]] = None
) -> Union[CandidateVolumeResponse, RequisitionNotFoundResponse]:
    """
    Retrieves candidate volume per sourcing channel.

    Args:
        requisition_id: The specific requisition ID to filter candidate volume.
        sources: Optional subset of sourcing channels to include (case-sensitive).

    Returns:
        A dictionary with source names and candidate volumes.
    """
    track_tool_call("candidate_source_candidate_volume_by_source")
    return get_candidate_volume_by_source(requisition_id, sources)


@mcp.tool()
def candidate_source_funnel_conversion_by_source(requisition_id: str) -> Union[FunnelConversionResponse, RequisitionNotFoundResponse]:
    """
    Retrieves conversion rates at each funnel stage for each sourcing channel.

    Args:
        requisition_id: The specific requisition ID to filter funnel data for.

    Returns:
        A dictionary with review %, interview rate, and offer acceptance rate.
    """
    track_tool_call("candidate_source_funnel_conversion_by_source")
    return get_funnel_conversion_by_source(requisition_id)


@mcp.tool()
def candidate_source_metadata_and_timeframe(requisition_id: str) -> Union[MetadataResponse, RequisitionNotFoundResponse]:
    """
    Retrieves metadata including data timeframe, last update date, and the
    number of requisitions analysed.

    Args:
        requisition_id: The job requisition ID.

    Returns:
        A dictionary containing timeframe and requisition summary.
    """
    track_tool_call("candidate_source_metadata_and_timeframe")
    return get_metadata_and_timeframe(requisition_id)


@mcp.tool()
def candidate_source_definitions_and_methodology(requisition_id: str) -> Union[DefinitionsResponse, RequisitionNotFoundResponse]:
    """
    Provides definitions of key metrics and outlines the methodology used
    to calculate performance.

    Args:
        requisition_id: The specific requisition ID for context.

    Returns:
        A dictionary including metric definitions, calculation notes,
        and the top metrics considered.
    """
    track_tool_call("candidate_source_definitions_and_methodology")
    return get_definitions_and_methodology(requisition_id)


@mcp.tool()
def candidate_source_source_recommendation_summary(requisition_id: str) -> Union[SourceRecommendationResponse, RequisitionNotFoundResponse]:
    """
    Returns a high-level summary combining jobs-filled %, review %, offer-accept
    rate, and total hires for each source.

    Args:
        requisition_id: The job requisition ID.

    Returns:
        A dictionary with composite source metrics.
    """
    track_tool_call("candidate_source_source_recommendation_summary")
    return get_source_recommendation_summary(requisition_id)


# Skills Tools
@mcp.tool()
def skills_skill_analysis(requisition_id: str) -> Union[SkillAnalysisResponse, RequisitionNotFoundResponse]:
    """
    Provides statistical indicators for each skill associated with the requisition,
    enabling an LLM or analyst to decide whether a skill should be retained,
    removed, or reconsidered.

    Args:
        requisition_id: The job requisition ID.

    Returns:
        Dict with historical counts and SLA correlation per skill.
    """
    track_tool_call("skills_skill_analysis")
    return get_skill_analysis(requisition_id)


@mcp.tool()
def skills_skill_impact_fill_rate(requisition_id: str, skill_name: str) -> Union[SkillImpactFillRateResponse, RequisitionNotFoundResponse]:
    """
    Evaluates how the inclusion of a specific skill affects requisition
    fill-rate metrics and candidate pool size.

    Args:
        requisition_id: The job requisition ID.
        skill_name: The skill to evaluate.

    Returns:
        Impact metrics with and without the skill.
    """
    track_tool_call("skills_skill_impact_fill_rate")
    return get_skill_impact_fill_rate(requisition_id, skill_name)


@mcp.tool()
def skills_skill_impact_sla(requisition_id: str, skill_name: str) -> Union[SkillImpactSLAResponse, RequisitionNotFoundResponse]:
    """
    Analyzes how a skill affects SLA achievement rate.

    Args:
        requisition_id: The job requisition ID.
        skill_name: The skill being analyzed.

    Returns:
        Success percentages with/without the skill and the delta.
    """
    track_tool_call("skills_skill_impact_sla")
    return get_skill_impact_sla(requisition_id, skill_name)


@mcp.tool()
def skills_skill_relevance_justification(requisition_id: str, skill_name: str) -> Union[SkillRelevanceResponse, RequisitionNotFoundResponse]:
    """
    Explains whether a skill is relevant and why, based on historical hiring
    success and outcome data.

    Args:
        requisition_id: The job requisition ID.
        skill_name: The skill being justified.

    Returns:
        Relevance determination with justification.
    """
    track_tool_call("skills_skill_relevance_justification")
    return get_skill_relevance_justification(requisition_id, skill_name)


@mcp.tool()
def skills_successful_posting_criteria() -> SuccessfulPostingResponse:
    """
    Returns the business definition of a successful job posting,
    including thresholds and benchmarks for success.

    Returns:
        Success criteria thresholds.
    """
    track_tool_call("skills_successful_posting_criteria")
    return get_successful_posting_criteria()


@mcp.tool()
def skills_data_sources_used(requisition_id: str) -> Union[DataSourcesResponse, RequisitionNotFoundResponse]:
    """
    Lists the datasets and ML models used to make hiring recommendations
    for a requisition.

    Args:
        requisition_id: The job requisition ID.

    Returns:
        Data sources and models used.
    """
    track_tool_call("skills_data_sources_used")
    return get_data_sources_used(requisition_id)


# ============================================================================
# Error-Prone Tools (for negative/hardness testing)
# ============================================================================

# --- Type Mismatch Tools ---

@mcp.tool()
def skills_skill_summary(requisition_id: str) -> str:
    """Get a quick text summary of skills needed for a requisition. Returns a concise skill overview."""
    track_tool_call("skills_skill_summary")
    return skills_error.get_skill_summary(requisition_id)


@mcp.tool()
def candidate_source_source_sla_score(requisition_id: str, source_name: str = "Dice"):
    """Get the SLA score for a specific sourcing channel. Returns the SLA achievement score."""
    track_tool_call("candidate_source_source_sla_score")
    return candidate_source_error.get_source_sla_score(requisition_id, source_name)


@mcp.tool()
def candidate_source_inactive_sources(requisition_id: str):
    """Show any inactive sourcing channels with no candidates."""
    track_tool_call("candidate_source_inactive_sources")
    return candidate_source_error.get_inactive_sources(requisition_id)


# --- HTTP Error Tools ---

@mcp.tool()
def candidate_source_candidate_pipeline_status(requisition_id: str):
    """Get candidate pipeline status showing distribution by source."""
    track_tool_call("candidate_source_candidate_pipeline_status")
    result = candidate_source_error.get_candidate_pipeline_status(requisition_id)
    if isinstance(result, dict) and result.get("error") and result.get("status_code"):
        raise Exception(json.dumps(result))
    return result


@mcp.tool()
def candidate_source_source_sla_check(requisition_id: str):
    """Run a quick SLA status check across all sourcing channels."""
    track_tool_call("candidate_source_source_sla_check")
    result = candidate_source_error.get_source_sla_check(requisition_id)
    if isinstance(result, dict) and result.get("error") and result.get("status_code", 0) >= 500:
        raise Exception(json.dumps(result))
    return result


@mcp.tool()
def candidate_source_funnel_status(requisition_id: str):
    """Get the current funnel status showing conversion at each stage."""
    track_tool_call("candidate_source_funnel_status")
    result = candidate_source_error.get_funnel_status(requisition_id)
    if isinstance(result, dict) and result.get("error"):
        raise Exception(json.dumps(result))
    return result


@mcp.tool()
def candidate_source_bulk_source_data(requisition_id: str):
    """Pull bulk source data for all requisitions in the system."""
    track_tool_call("candidate_source_bulk_source_data")
    result = candidate_source_error.get_bulk_source_data(requisition_id)
    if isinstance(result, dict) and result.get("error") and result.get("status_code") == 429:
        raise Exception(json.dumps(result))
    return result


# --- Schema Violation Tools ---

@mcp.tool()
def skills_model_registry(requisition_id: str):
    """Check which ML models are registered for a given requisition."""
    track_tool_call("skills_model_registry")
    return skills_error.get_model_registry(requisition_id)


@mcp.tool()
def skills_skill_lookup(requisition_id: str, skill_name: str = None):
    """Look up a specific skill and its metrics for a requisition."""
    track_tool_call("skills_skill_lookup")
    return skills_error.get_skill_lookup(requisition_id, skill_name)


@mcp.tool()
def candidate_source_source_metrics_lite(requisition_id: str):
    """Get a lightweight summary of source metrics for quick analysis."""
    track_tool_call("candidate_source_source_metrics_lite")
    return candidate_source_error.get_source_metrics_lite(requisition_id)


@mcp.tool()
def candidate_source_volume_report(requisition_id: str):
    """Generate a volume report showing candidate statistics by source."""
    track_tool_call("candidate_source_volume_report")
    return candidate_source_error.get_volume_report(requisition_id)


# --- Edge Case Tools ---

@mcp.tool()
def candidate_source_full_candidate_details(requisition_id: str):
    """Get full candidate-level details for comprehensive analysis."""
    track_tool_call("candidate_source_full_candidate_details")
    return candidate_source_error.get_full_candidate_details(requisition_id)


@mcp.tool()
def candidate_source_source_directory(requisition_id: str):
    """Show the source directory listing all sourcing channels with metadata."""
    track_tool_call("candidate_source_source_directory")
    return candidate_source_error.get_source_directory(requisition_id)


@mcp.tool()
def skills_skill_deep_analysis(requisition_id: str):
    """Get a deep analysis breakdown of skills with detailed sub-categories."""
    track_tool_call("skills_skill_deep_analysis")
    return skills_error.get_skill_deep_analysis(requisition_id)


@mcp.tool()
def candidate_source_sla_extended(requisition_id: str, source_name: str = "Dice"):
    """Get extended SLA data with additional analytics for a sourcing channel."""
    track_tool_call("candidate_source_sla_extended")
    return candidate_source_error.get_sla_extended(requisition_id, source_name)


@mcp.tool()
def skills_analyze_skill_match(requisition_id: str, skill_id: str):
    """Check if a skill is a good match for a requisition based on historical data."""
    track_tool_call("skills_analyze_skill_match")
    return skills_error.analyze_skill_match(requisition_id, skill_id)


# --- Undocumented Behavior Tools ---

@mcp.tool()
def candidate_source_requisition_details(requisition_id: str):
    """Get detailed information for a specific requisition."""
    track_tool_call("candidate_source_requisition_details")
    return candidate_source_error.get_requisition_details(requisition_id)


@mcp.tool()
def candidate_source_list_all_sources(requisition_id: str):
    """List all available sourcing channels in the system."""
    track_tool_call("candidate_source_list_all_sources")
    return candidate_source_error.list_all_sources(requisition_id)


@mcp.tool()
def candidate_source_batch_metrics(requisition_id: str):
    """Fetch aggregated batch metrics across all sourcing channels."""
    track_tool_call("candidate_source_batch_metrics")
    return candidate_source_error.get_batch_metrics(requisition_id)


# Tool registry for HTTP endpoint - uses underlying API functions with tracking
def _wrap_with_tracking(tool_name: str, func: callable) -> callable:
    """Wrap a function to track tool calls."""
    def wrapper(*args, **kwargs):
        track_tool_call(tool_name)
        return func(*args, **kwargs)
    return wrapper


TOOL_REGISTRY: Dict[str, callable] = {
    "candidate_source_sla_per_source": _wrap_with_tracking(
        "candidate_source_sla_per_source", get_sla_per_source),
    "candidate_source_total_hires_by_source": _wrap_with_tracking(
        "candidate_source_total_hires_by_source", get_total_hires_by_source),
    "candidate_source_candidate_volume_by_source": _wrap_with_tracking(
        "candidate_source_candidate_volume_by_source", get_candidate_volume_by_source),
    "candidate_source_funnel_conversion_by_source": _wrap_with_tracking(
        "candidate_source_funnel_conversion_by_source", get_funnel_conversion_by_source),
    "candidate_source_metadata_and_timeframe": _wrap_with_tracking(
        "candidate_source_metadata_and_timeframe", get_metadata_and_timeframe),
    "candidate_source_definitions_and_methodology": _wrap_with_tracking(
        "candidate_source_definitions_and_methodology", get_definitions_and_methodology),
    "candidate_source_source_recommendation_summary": _wrap_with_tracking(
        "candidate_source_source_recommendation_summary", get_source_recommendation_summary),
    "skills_skill_analysis": _wrap_with_tracking(
        "skills_skill_analysis", get_skill_analysis),
    "skills_skill_impact_fill_rate": _wrap_with_tracking(
        "skills_skill_impact_fill_rate", get_skill_impact_fill_rate),
    "skills_skill_impact_sla": _wrap_with_tracking(
        "skills_skill_impact_sla", get_skill_impact_sla),
    "skills_skill_relevance_justification": _wrap_with_tracking(
        "skills_skill_relevance_justification", get_skill_relevance_justification),
    "skills_successful_posting_criteria": _wrap_with_tracking(
        "skills_successful_posting_criteria", get_successful_posting_criteria),
    "skills_data_sources_used": _wrap_with_tracking(
        "skills_data_sources_used", get_data_sources_used),
    # Error-prone tools
    "skills_skill_summary": _wrap_with_tracking(
        "skills_skill_summary", skills_error.get_skill_summary),
    "candidate_source_source_sla_score": _wrap_with_tracking(
        "candidate_source_source_sla_score", candidate_source_error.get_source_sla_score),
    "candidate_source_inactive_sources": _wrap_with_tracking(
        "candidate_source_inactive_sources", candidate_source_error.get_inactive_sources),
    "candidate_source_candidate_pipeline_status": _wrap_with_tracking(
        "candidate_source_candidate_pipeline_status", candidate_source_error.get_candidate_pipeline_status),
    "candidate_source_source_sla_check": _wrap_with_tracking(
        "candidate_source_source_sla_check", candidate_source_error.get_source_sla_check),
    "candidate_source_funnel_status": _wrap_with_tracking(
        "candidate_source_funnel_status", candidate_source_error.get_funnel_status),
    "candidate_source_bulk_source_data": _wrap_with_tracking(
        "candidate_source_bulk_source_data", candidate_source_error.get_bulk_source_data),
    "skills_model_registry": _wrap_with_tracking(
        "skills_model_registry", skills_error.get_model_registry),
    "skills_skill_lookup": _wrap_with_tracking(
        "skills_skill_lookup", skills_error.get_skill_lookup),
    "candidate_source_source_metrics_lite": _wrap_with_tracking(
        "candidate_source_source_metrics_lite", candidate_source_error.get_source_metrics_lite),
    "candidate_source_volume_report": _wrap_with_tracking(
        "candidate_source_volume_report", candidate_source_error.get_volume_report),
    "candidate_source_full_candidate_details": _wrap_with_tracking(
        "candidate_source_full_candidate_details", candidate_source_error.get_full_candidate_details),
    "candidate_source_source_directory": _wrap_with_tracking(
        "candidate_source_source_directory", candidate_source_error.get_source_directory),
    "skills_skill_deep_analysis": _wrap_with_tracking(
        "skills_skill_deep_analysis", skills_error.get_skill_deep_analysis),
    "candidate_source_sla_extended": _wrap_with_tracking(
        "candidate_source_sla_extended", candidate_source_error.get_sla_extended),
    "skills_analyze_skill_match": _wrap_with_tracking(
        "skills_analyze_skill_match", skills_error.analyze_skill_match),
    "candidate_source_requisition_details": _wrap_with_tracking(
        "candidate_source_requisition_details", candidate_source_error.get_requisition_details),
    "candidate_source_list_all_sources": _wrap_with_tracking(
        "candidate_source_list_all_sources", candidate_source_error.list_all_sources),
    "candidate_source_batch_metrics": _wrap_with_tracking(
        "candidate_source_batch_metrics", candidate_source_error.get_batch_metrics),
}


class ToolCallRequest(BaseModel):
    """Request model for tool calls."""
    name: str
    arguments: Dict[str, Any] = {}


class ToolCallResponse(BaseModel):
    """Response model matching MCP format."""
    content: List[Dict[str, Any]]


def create_http_app() -> FastAPI:
    """Create FastAPI app with MCP tool call endpoint."""
    app = FastAPI(
        title="BPO Benchmark MCP Tools",
        description="HTTP API for BPO benchmark MCP tools",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/mcp/v1/tools/call", response_model=ToolCallResponse)
    async def call_tool(request: ToolCallRequest) -> ToolCallResponse:
        """Call an MCP tool and return the result."""
        tool_name = request.name
        arguments = request.arguments

        if tool_name not in TOOL_REGISTRY:
            raise HTTPException(
                status_code=404,
                detail=f"Tool not found: {tool_name}. Available tools: {list(TOOL_REGISTRY.keys())}"
            )

        tool_func = TOOL_REGISTRY[tool_name]
        logger.info(f"HTTP tool call: {tool_name}({arguments})")

        try:
            result = tool_func(**arguments)

            # Handle error-prone tool responses with HTTP status codes
            if isinstance(result, dict) and result.get("error") and result.get("status_code"):
                status = result["status_code"]
                if status in (500, 503, 429):
                    raise HTTPException(
                        status_code=status, detail=json.dumps(result)
                    )

            # Convert Pydantic models to dict if needed
            if hasattr(result, "model_dump"):
                result_dict = result.model_dump()
            elif hasattr(result, "dict"):
                result_dict = result.dict()
            else:
                result_dict = result

            return ToolCallResponse(
                content=[{"type": "text", "text": json.dumps(result_dict)}]
            )
        except TypeError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid arguments for tool {tool_name}: {e}"
            )
        except Exception as e:
            logger.error(f"Tool call failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Tool execution failed: {e}"
            )

    @app.get("/api/tools")
    async def list_tools() -> Dict[str, Any]:
        """List all available tools."""
        return {
            "server": "bpo_benchmark",
            "total_tools": len(TOOL_REGISTRY),
            "tools": list(TOOL_REGISTRY.keys())
        }

    @app.get("/health")
    async def health_check() -> Dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy"}

    return app


def start_server(port: int = None, host: str = "0.0.0.0", transport: str = "stdio"):
    """
    Start the MCP server.

    Args:
        port: Port to listen on (default: None for stdio)
        host: Host to bind to (default: 0.0.0.0)
        transport: Transport mode - "stdio", "http", or "sse" (default: stdio)
    """
    logger.info(f"Starting MCP server (transport={transport}, port={port}, host={host})")

    if transport == "http" and port:
        # Run with custom HTTP server (FastAPI with REST endpoint)
        logger.info(f"Starting HTTP server on http://{host}:{port}")
        logger.info(f"Tool call endpoint: http://{host}:{port}/mcp/v1/tools/call")
        logger.info(f"API docs: http://{host}:{port}/docs")
        app = create_http_app()
        uvicorn.run(app, host=host, port=port)
    elif transport in ("sse", "streamable-http") and port:
        # Run with FastMCP SSE/streamable-http transport
        import asyncio
        from fastmcp_docs import FastMCPDocs

        # Setup documentation UI
        logger.info("Setting up documentation UI at /docs")
        docs = FastMCPDocs(mcp, title="BPO Benchmark MCP Tools")
        asyncio.run(docs.setup())

        # Start server
        logger.info(f"Starting {transport.upper()} server on http://{host}:{port}")
        logger.info(f"Documentation available at http://{host}:{port}/docs")
        mcp.run(transport=transport, host=host, port=port)
    else:
        # Run with default FastMCP mode (stdio)
        logger.info("Starting server in stdio mode")
        mcp.run()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Start BPO MCP Server")
    parser.add_argument("--port", type=int, help="Port for HTTP/SSE mode (default: stdio)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    start_server(port=args.port, host=args.host)
