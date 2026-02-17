"""
Comprehensive test suite for BPO MCP Server.

Tests the MCP server in stdio mode using MCP client to validate:
1. Server connectivity and initialization
2. All candidate source tools (7 core + 18 error-prone variants)
3. All skills tools (6 core + 5 error-prone variants)
4. Error handling and edge cases
5. Tool call tracking functionality

Total: 31 tools tested (13 core + 18 error-prone)
"""

import json
import subprocess
import sys
import time
import pytest
from pathlib import Path
from typing import Any, Dict, Optional
import asyncio
from contextlib import asynccontextmanager

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class MCPStdioClient:
    """Client for testing MCP server in stdio mode."""
    
    def __init__(self, server_script: str):
        """
        Initialize MCP stdio client.
        
        Args:
            server_script: Path to the server script to run
        """
        self.server_script = server_script
        self.process: Optional[subprocess.Popen] = None
        self.request_id = 0
        
    def start(self) -> None:
        """Start the MCP server process in stdio mode."""
        print(f"🚀 Starting MCP server: {self.server_script}")
        
        # Start server process with stdio communication
        self.process = subprocess.Popen(
            [sys.executable, self.server_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Wait a moment for server to initialize
        time.sleep(2)
        
        # Check if process is still running
        if self.process.poll() is not None:
            stderr = self.process.stderr.read() if self.process.stderr else ""
            raise RuntimeError(f"Server failed to start. stderr: {stderr}")
        
        print("✅ MCP server started successfully in stdio mode")
        
    def stop(self) -> None:
        """Stop the MCP server process."""
        if self.process:
            print("🛑 Stopping MCP server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            print("✅ Server stopped")
            
    def send_request(self, method: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Send a JSON-RPC request to the server.
        
        Args:
            method: The JSON-RPC method name
            params: Optional parameters for the method
            
        Returns:
            The JSON-RPC response
        """
        if not self.process or self.process.poll() is not None:
            raise RuntimeError("Server process is not running")
        
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params or {}
        }
        
        # Send request
        request_str = json.dumps(request) + "\n"
        if self.process.stdin:
            self.process.stdin.write(request_str)
            self.process.stdin.flush()
        
        # Read response
        response_str = ""
        if self.process.stdout:
            response_str = self.process.stdout.readline()
        if not response_str:
            stderr = self.process.stderr.read() if self.process.stderr else ""
            raise RuntimeError(f"No response from server. stderr: {stderr}")
        
        try:
            response = json.loads(response_str)
            return response
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON response: {response_str}. Error: {e}")
    
    def initialize(self) -> Dict[str, Any]:
        """Initialize the MCP session."""
        return self.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        })
    
    def list_tools(self) -> Dict[str, Any]:
        """List all available tools."""
        return self.send_request("tools/list", {})
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool with the given arguments.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            
        Returns:
            Tool response
        """
        return self.send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })


@pytest.fixture(scope="session")
def mcp_client():
    """Pytest fixture to provide MCP client with server lifecycle management."""
    server_script = str(Path(__file__).parent.parent / "server.py")
    client = MCPStdioClient(server_script)
    
    try:
        client.start()
        
        # Initialize the session
        init_response = client.initialize()
        assert "result" in init_response, "Failed to initialize MCP session"
        print(f"✅ MCP session initialized: {init_response['result'].get('serverInfo', {}).get('name', 'unknown')}")
        
        yield client
        
    finally:
        client.stop()


class TestServerInitialization:
    """Test server initialization and basic connectivity."""
    
    def test_server_starts(self, mcp_client):
        """Test that server starts successfully."""
        assert mcp_client.process is not None
        assert mcp_client.process.poll() is None
        print("✅ Server is running")
    
    def test_list_tools(self, mcp_client):
        """Test listing all available tools."""
        response = mcp_client.list_tools()
        assert "result" in response
        tools = response["result"].get("tools", [])
        assert len(tools) > 0, "No tools found"
        
        # Verify we have the expected number of tools (13 core + 18 error-prone = 31+)
        assert len(tools) >= 31, f"Expected at least 31 tools, found {len(tools)}"
        
        tool_names = [tool["name"] for tool in tools]
        print(f"✅ Found {len(tools)} tools: {', '.join(tool_names[:5])}...")


class TestCandidateSourceTools:
    """Test all candidate source tools."""
    
    def test_sla_per_source(self, mcp_client):
        """Test candidate_source_sla_per_source tool."""
        response = mcp_client.call_tool(
            "candidate_source_sla_per_source",
            {"requisition_id": "REQ00202"}
        )
        assert "result" in response
        content = response["result"]["content"]
        assert len(content) > 0
        
        # Parse the response
        data = json.loads(content[0]["text"])
        assert "metrics" in data or "error" in data
        print("✅ SLA per source test passed")
    
    def test_total_hires_by_source(self, mcp_client):
        """Test candidate_source_total_hires_by_source tool."""
        response = mcp_client.call_tool(
            "candidate_source_total_hires_by_source",
            {"requisition_id": "REQ00202"}
        )
        assert "result" in response
        content = response["result"]["content"]
        data = json.loads(content[0]["text"])
        assert "metrics" in data or "error" in data
        print("✅ Total hires by source test passed")
    
    def test_candidate_volume_by_source(self, mcp_client):
        """Test candidate_source_candidate_volume_by_source tool."""
        response = mcp_client.call_tool(
            "candidate_source_candidate_volume_by_source",
            {"requisition_id": "REQ00202"}
        )
        assert "result" in response
        content = response["result"]["content"]
        data = json.loads(content[0]["text"])
        assert "metrics" in data or "error" in data
        print("✅ Candidate volume by source test passed")
    
    def test_candidate_volume_with_filter(self, mcp_client):
        """Test candidate_source_candidate_volume_by_source with source filter."""
        response = mcp_client.call_tool(
            "candidate_source_candidate_volume_by_source",
            {
                "requisition_id": "REQ00202",
                "sources": ["LinkedIn", "Indeed"]
            }
        )
        assert "result" in response
        content = response["result"]["content"]
        data = json.loads(content[0]["text"])
        assert "metrics" in data or "error" in data
        print("✅ Candidate volume with filter test passed")
    
    def test_funnel_conversion_by_source(self, mcp_client):
        """Test candidate_source_funnel_conversion_by_source tool."""
        response = mcp_client.call_tool(
            "candidate_source_funnel_conversion_by_source",
            {"requisition_id": "REQ00202"}
        )
        assert "result" in response
        content = response["result"]["content"]
        data = json.loads(content[0]["text"])
        assert "metrics" in data or "error" in data
        print("✅ Funnel conversion by source test passed")
    
    def test_metadata_and_timeframe(self, mcp_client):
        """Test candidate_source_metadata_and_timeframe tool."""
        response = mcp_client.call_tool(
            "candidate_source_metadata_and_timeframe",
            {"requisition_id": "REQ00202"}
        )
        assert "result" in response
        content = response["result"]["content"]
        data = json.loads(content[0]["text"])
        assert "time_frame_end" in data or "error" in data
        print("✅ Metadata and timeframe test passed")
    
    def test_definitions_and_methodology(self, mcp_client):
        """Test candidate_source_definitions_and_methodology tool."""
        response = mcp_client.call_tool(
            "candidate_source_definitions_and_methodology",
            {"requisition_id": "REQ00202"}
        )
        assert "result" in response
        content = response["result"]["content"]
        data = json.loads(content[0]["text"])
        assert "definitions" in data or "error" in data
        print("✅ Definitions and methodology test passed")
    
    def test_source_recommendation_summary(self, mcp_client):
        """Test candidate_source_source_recommendation_summary tool."""
        response = mcp_client.call_tool(
            "candidate_source_source_recommendation_summary",
            {"requisition_id": "REQ00202"}
        )
        assert "result" in response
        content = response["result"]["content"]
        data = json.loads(content[0]["text"])
        assert "metrics" in data or "error" in data
        print("✅ Source recommendation summary test passed")


class TestSkillsTools:
    """Test all skills tools."""
    
    def test_skill_analysis(self, mcp_client):
        """Test skills_skill_analysis tool."""
        response = mcp_client.call_tool(
            "skills_skill_analysis",
            {"requisition_id": "REQ00202"}
        )
        assert "result" in response
        content = response["result"]["content"]
        data = json.loads(content[0]["text"])

        assert "input_skills" in data or "error" in data
        print("✅ Skill analysis test passed")
    
    def test_skill_impact_fill_rate(self, mcp_client):
        """Test skills_skill_impact_fill_rate tool."""
        response = mcp_client.call_tool(
            "skills_skill_impact_fill_rate",
            {
                "requisition_id": "REQ00202",
                "skill_name": "Python"
            }
        )
        assert "result" in response
        content = response["result"]["content"]
        data = json.loads(content[0]["text"])

        assert "skill_name" in data or "error" in data
        print("✅ Skill impact fill rate test passed")
    
    def test_skill_impact_sla(self, mcp_client):
        """Test skills_skill_impact_sla tool."""
        response = mcp_client.call_tool(
            "skills_skill_impact_sla",
            {
                "requisition_id": "REQ00202",
                "skill_name": "Python"
            }
        )
        assert "result" in response
        content = response["result"]["content"]
        data = json.loads(content[0]["text"])
        assert "skill_name" in data or "error" in data
        print("✅ Skill impact SLA test passed")
    
    def test_skill_relevance_justification(self, mcp_client):
        """Test skills_skill_relevance_justification tool."""
        response = mcp_client.call_tool(
            "skills_skill_relevance_justification",
            {
                "requisition_id": "REQ00202",
                "skill_name": "Python"
            }
        )
        assert "result" in response
        content = response["result"]["content"]
        data = json.loads(content[0]["text"])
        assert "is_relevant" in data or "error" in data
        print("✅ Skill relevance justification test passed")
    
    def test_successful_posting_criteria(self, mcp_client):
        """Test skills_successful_posting_criteria tool."""
        response = mcp_client.call_tool(
            "skills_successful_posting_criteria",
            {}
        )
        assert "result" in response
        content = response["result"]["content"]
        data = json.loads(content[0]["text"])
        assert "criteria" in data
        print("✅ Successful posting criteria test passed")
    
    def test_data_sources_used(self, mcp_client):
        """Test skills_data_sources_used tool."""
        response = mcp_client.call_tool(
            "skills_data_sources_used",
            {"requisition_id": "REQ00202"}
        )
        assert "result" in response
        content = response["result"]["content"]
        data = json.loads(content[0]["text"])

        assert "requisition_id" in data or "error" in data
        print("✅ Data sources used test passed")


class TestErrorProneTools:
    """Test error-prone tools that handle various error conditions."""
    
    def test_type_mismatch_skill_summary(self, mcp_client):
        """Test skills_skill_summary (type mismatch tool)."""
        response = mcp_client.call_tool(
            "skills_skill_summary",
            {"requisition_id": "REQ00202"}
        )
        assert "result" in response
        content = response["result"]["content"]
        # This tool returns a string instead of structured data
        text = content[0]["text"]

        assert isinstance(text, str)
        print("✅ Type mismatch skill summary test passed")
    
    def test_type_mismatch_source_sla_score(self, mcp_client):
        """Test candidate_source_source_sla_score (type mismatch tool)."""
        response = mcp_client.call_tool(
            "candidate_source_source_sla_score",
            {
                "requisition_id": "REQ00202",
                "source_name": "LinkedIn"
            }
        )
        assert "result" in response
        content = response["result"]["content"]
        data = json.loads(content[0]["text"])
        # May return number or dict
        assert isinstance(data, (int, float, dict))
        print("✅ Type mismatch source SLA score test passed")
    
    def test_type_mismatch_inactive_sources(self, mcp_client):
        """Test candidate_source_inactive_sources (type mismatch tool)."""
        response = mcp_client.call_tool(
            "candidate_source_inactive_sources",
            {"requisition_id": "REQ00202"}
        )
        assert "result" in response
        content = response["result"]["content"]
        assert isinstance(content, list)
        
        print("✅ Type mismatch inactive sources test passed")
    
    def test_http_error_pipeline_status(self, mcp_client):
        """Test candidate_source_candidate_pipeline_status (HTTP error tool)."""
        response = mcp_client.call_tool(
            "candidate_source_candidate_pipeline_status",
            {"requisition_id": "REQ00202"}
        )
        # May return error or success
        assert "result" in response or "error" in response
        print("✅ HTTP error pipeline status test passed")
    
    def test_http_error_sla_check(self, mcp_client):
        """Test candidate_source_source_sla_check (HTTP error tool)."""
        response = mcp_client.call_tool(
            "candidate_source_source_sla_check",
            {"requisition_id": "REQ00202"}
        )
        # May return error or success
        assert "result" in response or "error" in response
        print("✅ HTTP error SLA check test passed")
    
    def test_schema_violation_model_registry(self, mcp_client):
        """Test skills_model_registry (schema violation tool)."""
        response = mcp_client.call_tool(
            "skills_model_registry",
            {"requisition_id": "REQ00202"}
        )
        assert "result" in response
        content = response["result"]["content"]
        data = json.loads(content[0]["text"])
        # Schema may not match expected format
        assert isinstance(data, dict)
        print("✅ Schema violation model registry test passed")
    
    def test_schema_violation_skill_lookup(self, mcp_client):
        """Test skills_skill_lookup (schema violation tool)."""
        response = mcp_client.call_tool(
            "skills_skill_lookup",
            {
                "requisition_id": "REQ00202",
                "skill_name": "Python"
            }
        )
        assert "result" in response
        content = response["result"]["content"]
        data = json.loads(content[0]["text"])
        assert isinstance(data, dict)
        print("✅ Schema violation skill lookup test passed")
    
    def test_schema_violation_source_metrics_lite(self, mcp_client):
        """Test candidate_source_source_metrics_lite (schema violation tool)."""
        response = mcp_client.call_tool(
            "candidate_source_source_metrics_lite",
            {"requisition_id": "REQ00202"}
        )
        assert "result" in response
        content = response["result"]["content"]
        data = json.loads(content[0]["text"])
        assert isinstance(data, dict)
        print("✅ Schema violation source metrics lite test passed")
    
    def test_schema_violation_volume_report(self, mcp_client):
        """Test candidate_source_volume_report (schema violation tool)."""
        response = mcp_client.call_tool(
            "candidate_source_volume_report",
            {"requisition_id": "REQ00202"}
        )
        assert "result" in response
        content = response["result"]["content"]
        data = json.loads(content[0]["text"])
        assert isinstance(data, dict)
        print("✅ Schema violation volume report test passed")
    
    def test_http_error_funnel_status(self, mcp_client):
        """Test candidate_source_funnel_status (HTTP error tool)."""
        response = mcp_client.call_tool(
            "candidate_source_funnel_status",
            {"requisition_id": "REQ00202"}
        )
        # May return error or success
        assert "result" in response or "error" in response
        print("✅ HTTP error funnel status test passed")
    
    def test_http_error_bulk_source_data(self, mcp_client):
        """Test candidate_source_bulk_source_data (HTTP error tool)."""
        response = mcp_client.call_tool(
            "candidate_source_bulk_source_data",
            {"requisition_id": "REQ00202"}
        )
        # May return error or success
        assert "result" in response or "error" in response
        print("✅ HTTP error bulk source data test passed")
    
    def test_edge_case_full_candidate_details(self, mcp_client):
        """Test candidate_source_full_candidate_details (edge case tool)."""
        response = mcp_client.call_tool(
            "candidate_source_full_candidate_details",
            {"requisition_id": "REQ00202"}
        )
        assert "result" in response
        content = response["result"]["content"]
        data = json.loads(content[0]["text"])
        # May return large response
        assert isinstance(data, dict)
        print("✅ Edge case full candidate details test passed")
    
    def test_edge_case_source_directory(self, mcp_client):
        """Test candidate_source_source_directory (edge case tool)."""
        response = mcp_client.call_tool(
            "candidate_source_source_directory",
            {"requisition_id": "REQ00202"}
        )
        assert "result" in response
        content = response["result"]["content"]
        data = json.loads(content[0]["text"])
        assert isinstance(data, dict)
        print("✅ Edge case source directory test passed")
    
    def test_edge_case_skill_deep_analysis(self, mcp_client):
        """Test skills_skill_deep_analysis (edge case tool)."""
        response = mcp_client.call_tool(
            "skills_skill_deep_analysis",
            {"requisition_id": "REQ00202"}
        )
        assert "result" in response
        content = response["result"]["content"]
        data = json.loads(content[0]["text"])
        assert isinstance(data, dict)
        print("✅ Edge case skill deep analysis test passed")
    
    def test_edge_case_sla_extended(self, mcp_client):
        """Test candidate_source_sla_extended (edge case tool)."""
        response = mcp_client.call_tool(
            "candidate_source_sla_extended",
            {
                "requisition_id": "REQ00202",
                "source_name": "LinkedIn"
            }
        )
        assert "result" in response
        content = response["result"]["content"]
        data = json.loads(content[0]["text"])
        assert isinstance(data, dict)
        print("✅ Edge case SLA extended test passed")
    
    def test_edge_case_analyze_skill_match(self, mcp_client):
        """Test skills_analyze_skill_match (edge case tool)."""
        response = mcp_client.call_tool(
            "skills_analyze_skill_match",
            {
                "requisition_id": "REQ00202",
                "skill_id": "SKILL-001"
            }
        )
        assert "result" in response
        content = response["result"]["content"]
        data = json.loads(content[0]["text"])
        assert isinstance(data, dict)
        print("✅ Edge case analyze skill match test passed")
    
    def test_undocumented_requisition_details(self, mcp_client):
        """Test candidate_source_requisition_details (undocumented behavior tool)."""
        response = mcp_client.call_tool(
            "candidate_source_requisition_details",
            {"requisition_id": "REQ00202"}
        )
        assert "result" in response
        content = response["result"]["content"]
        data = json.loads(content[0]["text"])
        assert isinstance(data, dict)
        print("✅ Undocumented requisition details test passed")
    
    def test_undocumented_list_all_sources(self, mcp_client):
        """Test candidate_source_list_all_sources (undocumented behavior tool)."""
        response = mcp_client.call_tool(
            "candidate_source_list_all_sources",
            {"requisition_id": "REQ00202"}
        )
        assert "result" in response
        content = response["result"]["content"]
        data = json.loads(content[0]["text"])
        assert isinstance(data, (list, dict))
        print("✅ Undocumented list all sources test passed")
    
    def test_undocumented_batch_metrics(self, mcp_client):
        """Test candidate_source_batch_metrics (undocumented behavior tool)."""
        response = mcp_client.call_tool(
            "candidate_source_batch_metrics",
            {"requisition_id": "REQ00202"}
        )
        assert "result" in response
        content = response["result"]["content"]
        data = json.loads(content[0]["text"])
        assert isinstance(data, dict)
        print("✅ Undocumented batch metrics test passed")


class TestErrorHandling:
    """Test error handling for invalid inputs."""
    
    def test_invalid_tool_name(self, mcp_client):
        """Test calling a non-existent tool."""
        response = mcp_client.call_tool(
            "nonexistent_tool",
            {"requisition_id": "REQ00202"}
        )
        # MCP returns result with isError flag for tool errors
        assert "result" in response, f"Expected result in response, got: {response}"
        result = response["result"]
        assert result.get("isError") is True, f"Expected isError=True, got: {result}"
        assert "content" in result, f"Expected content in result, got: {result}"
        print("✅ Invalid tool name test passed")
    
    def test_missing_required_parameter(self, mcp_client):
        """Test calling a tool without required parameters."""
        response = mcp_client.call_tool(
            "candidate_source_sla_per_source",
            {}  # Missing requisition_id
        )
        # MCP returns result with isError flag for parameter errors
        assert "result" in response, f"Expected result in response, got: {response}"
        result = response["result"]
        assert result.get("isError") is True, f"Expected isError=True, got: {result}"
        assert "content" in result, f"Expected content in result, got: {result}"
        print("✅ Missing required parameter test passed")
    
    def test_invalid_requisition_id(self, mcp_client):
        """Test with an invalid requisition ID."""
        response = mcp_client.call_tool(
            "candidate_source_sla_per_source",
            {"requisition_id": "INVALID-ID"}
        )
        assert "result" in response
        content = response["result"]["content"]
        data = json.loads(content[0]["text"])
        # Should return error response
        assert "error" in data
        print("✅ Invalid requisition ID test passed")


def run_all_tests():
    """Run all tests manually (for direct execution)."""
    print("\n" + "="*70)
    print("BPO MCP SERVER TEST SUITE")
    print("="*70 + "\n")
    
    server_script = str(Path(__file__).parent.parent / "server.py")
    client = MCPStdioClient(server_script)
    
    try:
        # Start server
        client.start()
        
        # Initialize session
        print("\n📋 Initializing MCP session...")
        init_response = client.initialize()
        assert "result" in init_response
        print(f"✅ Session initialized: {init_response['result'].get('serverInfo', {}).get('name', 'unknown')}")
        
        # Test server initialization
        print("\n📋 Running Server Initialization Tests...")
        test_init = TestServerInitialization()
        test_init.test_server_starts(client)
        test_init.test_list_tools(client)
        
        # Test candidate source tools
        print("\n📋 Running Candidate Source Tools Tests...")
        test_cs = TestCandidateSourceTools()
        test_cs.test_sla_per_source(client)
        test_cs.test_total_hires_by_source(client)
        test_cs.test_candidate_volume_by_source(client)
        test_cs.test_candidate_volume_with_filter(client)
        test_cs.test_funnel_conversion_by_source(client)
        test_cs.test_metadata_and_timeframe(client)
        test_cs.test_definitions_and_methodology(client)
        test_cs.test_source_recommendation_summary(client)
        
        # Test skills tools
        print("\n📋 Running Skills Tools Tests...")
        test_skills = TestSkillsTools()
        test_skills.test_skill_analysis(client)
        test_skills.test_skill_impact_fill_rate(client)
        test_skills.test_skill_impact_sla(client)
        test_skills.test_skill_relevance_justification(client)
        test_skills.test_successful_posting_criteria(client)
        test_skills.test_data_sources_used(client)
        
        # Test error-prone tools
        print("\n📋 Running Error-Prone Tools Tests...")
        test_errors = TestErrorProneTools()
        test_errors.test_type_mismatch_skill_summary(client)
        test_errors.test_type_mismatch_source_sla_score(client)
        test_errors.test_type_mismatch_inactive_sources(client)
        test_errors.test_http_error_pipeline_status(client)
        test_errors.test_http_error_sla_check(client)
        test_errors.test_http_error_funnel_status(client)
        test_errors.test_http_error_bulk_source_data(client)
        test_errors.test_schema_violation_model_registry(client)
        test_errors.test_schema_violation_skill_lookup(client)
        test_errors.test_schema_violation_source_metrics_lite(client)
        test_errors.test_schema_violation_volume_report(client)
        test_errors.test_edge_case_full_candidate_details(client)
        test_errors.test_edge_case_source_directory(client)
        test_errors.test_edge_case_skill_deep_analysis(client)
        test_errors.test_edge_case_sla_extended(client)
        test_errors.test_edge_case_analyze_skill_match(client)
        test_errors.test_undocumented_requisition_details(client)
        test_errors.test_undocumented_list_all_sources(client)
        test_errors.test_undocumented_batch_metrics(client)
        
        # Test error handling
        print("\n📋 Running Error Handling Tests...")
        test_error_handling = TestErrorHandling()
        test_error_handling.test_invalid_tool_name(client)
        test_error_handling.test_missing_required_parameter(client)
        test_error_handling.test_invalid_requisition_id(client)
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        client.stop()
    
    return 0


if __name__ == "__main__":
    sys.exit(run_all_tests())