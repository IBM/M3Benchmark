"""Integration tests for unified MCPToolWrapper.

Tests cover:
- Schema preservation (JSON Schema → Pydantic conversion)
- Tool conversion and caching
- OpenAI name sanitization
- Profiling functionality
- Regression compatibility
"""

import pytest
from unittest.mock import AsyncMock, Mock
from typing import Any

from agents.mcp_tool_wrapper import MCPToolWrapper


@pytest.fixture
def mock_mcp_session():
    """Create mock MCP session with sample tools."""
    session = AsyncMock()

    # Mock tool with comprehensive schema
    mock_tool = Mock()
    mock_tool.name = "filter_data"
    mock_tool.description = "Apply a filter based on a condition"
    mock_tool.inputSchema = {
        "type": "object",
        "properties": {
            "data": {
                "type": "object",
                "description": "Data to filter"
            },
            "key_name": {
                "type": "string",
                "description": "Filter key"
            },
            "value": {
                "type": "string",
                "description": "Filter value"
            },
            "condition": {
                "type": "string",
                "enum": ["equal_to", "not_equal_to", "greater_than"],
                "description": "Filter condition"
            }
        },
        "required": ["data", "key_name", "value", "condition"]
    }

    session.list_tools.return_value = Mock(tools=[mock_tool])
    session.call_tool.return_value = Mock(content=[Mock(text="result")])

    return session


@pytest.fixture
def mock_tool_with_complex_types():
    """Mock tool with complex type definitions."""
    tool = Mock()
    tool.name = "complex_tool"
    tool.description = "Tool with complex types"
    tool.inputSchema = {
        "type": "object",
        "properties": {
            "integer_field": {"type": "integer", "description": "An integer"},
            "number_field": {"type": "number", "description": "A float"},
            "boolean_field": {"type": "boolean", "description": "A boolean"},
            "array_field": {"type": "array", "description": "An array"},
            "object_field": {"type": "object", "description": "An object"},
            "union_field": {"anyOf": [{"type": "string"}, {"type": "integer"}], "description": "Union type"},
            "optional_field": {"type": "string", "description": "Optional field", "default": "default_value"}
        },
        "required": ["integer_field", "number_field", "boolean_field"]
    }
    return tool


# ============================================================================
# Schema Preservation Tests
# ============================================================================

@pytest.mark.asyncio
async def test_schema_conversion_basic_types(mock_mcp_session):
    """Test JSON Schema → Pydantic conversion for basic types."""
    wrapper = MCPToolWrapper(session=mock_mcp_session)
    tools = await wrapper.get_tools()

    assert len(tools) == 1
    tool = tools[0]

    # CRITICAL: Verify args_schema is not None
    assert tool.args_schema is not None, "args_schema must not be None for LLM visibility"

    # Verify schema has correct fields
    schema_dict = tool.args_schema.model_json_schema()
    properties = schema_dict["properties"]

    assert "data" in properties
    assert "key_name" in properties
    assert "value" in properties
    assert "condition" in properties

    # Verify types
    assert properties["key_name"]["type"] == "string"

    # Verify descriptions
    assert "Data to filter" in properties["data"]["description"]
    assert "Filter key" in properties["key_name"]["description"]


@pytest.mark.asyncio
async def test_schema_conversion_complex_types(mock_tool_with_complex_types, mock_mcp_session):
    """Test conversion for array, object, union types."""
    # Setup session with complex tool
    mock_mcp_session.list_tools.return_value = Mock(tools=[mock_tool_with_complex_types])

    wrapper = MCPToolWrapper(session=mock_mcp_session)
    tools = await wrapper.get_tools()

    tool = tools[0]
    assert tool.args_schema is not None

    schema_dict = tool.args_schema.model_json_schema()
    properties = schema_dict["properties"]

    # Verify complex type mappings
    assert "integer_field" in properties
    assert "number_field" in properties
    assert "boolean_field" in properties
    assert "array_field" in properties
    assert "object_field" in properties
    assert "union_field" in properties


@pytest.mark.asyncio
async def test_schema_conversion_enums(mock_mcp_session):
    """Test enum constraints are preserved in descriptions."""
    wrapper = MCPToolWrapper(session=mock_mcp_session)
    tools = await wrapper.get_tools()

    tool = tools[0]
    schema_dict = tool.args_schema.model_json_schema()
    properties = schema_dict["properties"]

    # Enum values should be in description
    condition_desc = properties["condition"]["description"]
    assert "equal_to" in condition_desc
    assert "not_equal_to" in condition_desc
    assert "greater_than" in condition_desc
    assert "Allowed values:" in condition_desc


@pytest.mark.asyncio
async def test_schema_conversion_required_optional(mock_tool_with_complex_types, mock_mcp_session):
    """Test required fields use Field(...), optional use Field(default=...)."""
    # Setup session with complex tool
    mock_mcp_session.list_tools.return_value = Mock(tools=[mock_tool_with_complex_types])

    wrapper = MCPToolWrapper(session=mock_mcp_session)
    tools = await wrapper.get_tools()

    tool = tools[0]
    schema_dict = tool.args_schema.model_json_schema()

    # Verify required fields
    required = set(schema_dict.get("required", []))
    assert "integer_field" in required
    assert "number_field" in required
    assert "boolean_field" in required

    # Optional fields should not be in required list
    assert "optional_field" not in required
    # union_field has no default, so it IS required
    assert "union_field" in required


@pytest.mark.asyncio
async def test_pydantic_model_validation(mock_mcp_session):
    """Test that generated Pydantic model validates inputs correctly."""
    wrapper = MCPToolWrapper(session=mock_mcp_session)
    tools = await wrapper.get_tools()

    tool = tools[0]
    pydantic_model = tool.args_schema

    # Valid input should pass
    valid_input = {
        "data": {"id": 1},
        "key_name": "id",
        "value": "1",
        "condition": "equal_to"
    }
    validated = pydantic_model(**valid_input)
    assert validated.key_name == "id"

    # Invalid input (missing required field) should raise
    with pytest.raises(Exception):  # Pydantic ValidationError
        pydantic_model(key_name="id", value="1", condition="equal_to")


@pytest.mark.asyncio
async def test_llm_sees_correct_schema(mock_mcp_session):
    """Test that LLM receives complete tool schema via args_schema."""
    wrapper = MCPToolWrapper(session=mock_mcp_session)
    tools = await wrapper.get_tools()

    tool = tools[0]

    # This is what LangChain will use to show the tool schema to the LLM
    assert tool.args_schema is not None

    # Verify the schema can be serialized (this is what LLM sees)
    schema_dict = tool.args_schema.model_json_schema()

    # Verify all essential schema components are present
    assert "properties" in schema_dict
    assert "required" in schema_dict
    assert len(schema_dict["properties"]) == 4
    assert len(schema_dict["required"]) == 4


# ============================================================================
# Tool Conversion Tests
# ============================================================================

@pytest.mark.asyncio
async def test_get_tools_from_mcp_session(mock_mcp_session):
    """Test fetching and converting tools from MCP session."""
    wrapper = MCPToolWrapper(session=mock_mcp_session)
    tools = await wrapper.get_tools()

    # Verify list_tools was called
    mock_mcp_session.list_tools.assert_called_once()

    # Verify tools converted correctly
    assert len(tools) == 1
    assert tools[0].name == "filter_data"
    assert "filter" in tools[0].description.lower()


@pytest.mark.asyncio
async def test_tool_caching(mock_mcp_session):
    """Test that tools are cached and not refetched."""
    wrapper = MCPToolWrapper(session=mock_mcp_session, cache_tools=True)

    # First call
    tools1 = await wrapper.get_tools()

    # Second call
    tools2 = await wrapper.get_tools()

    # list_tools should only be called once (caching works)
    assert mock_mcp_session.list_tools.call_count == 1

    # Same tools returned
    assert tools1 is tools2


@pytest.mark.asyncio
async def test_no_caching_when_disabled(mock_mcp_session):
    """Test that caching can be disabled."""
    wrapper = MCPToolWrapper(session=mock_mcp_session, cache_tools=False)

    # First call
    await wrapper.get_tools()

    # Second call
    await wrapper.get_tools()

    # list_tools should be called twice (no caching)
    assert mock_mcp_session.list_tools.call_count == 2


@pytest.mark.asyncio
async def test_tool_function_execution(mock_mcp_session):
    """Test that tool functions call session.call_tool correctly."""
    wrapper = MCPToolWrapper(session=mock_mcp_session)
    tools = await wrapper.get_tools()

    tool = tools[0]

    # Execute tool via coroutine
    result = await tool.coroutine(
        data={"id": 1},
        key_name="id",
        value="1",
        condition="equal_to"
    )

    # Verify call_tool was invoked
    mock_mcp_session.call_tool.assert_called_once()
    call_args = mock_mcp_session.call_tool.call_args

    assert call_args[0][0] == "filter_data"  # tool name
    assert "data" in call_args[0][1]  # kwargs
    assert "key_name" in call_args[0][1]

    # Verify result
    assert result == "result"


@pytest.mark.asyncio
async def test_async_and_sync_tool_execution(mock_mcp_session):
    """Test both coroutine and func paths work."""
    wrapper = MCPToolWrapper(session=mock_mcp_session)
    tools = await wrapper.get_tools()

    tool = tools[0]

    # Test async execution
    async_result = await tool.coroutine(
        data={"id": 1},
        key_name="id",
        value="1",
        condition="equal_to"
    )
    assert async_result == "result"

    # Note: Sync execution test skipped when running in async context
    # The sync wrapper uses asyncio.run() which doesn't work within existing event loop
    # In real usage, the sync wrapper is for non-async contexts


# ============================================================================
# OpenAI Compatibility Tests
# ============================================================================

@pytest.mark.asyncio
async def test_openai_name_sanitization():
    """Test invalid characters replaced with underscores."""
    session = AsyncMock()

    tool = Mock()
    tool.name = "filter/data:v1@test"
    tool.description = "Test tool"
    tool.inputSchema = {"type": "object", "properties": {}}

    session.list_tools.return_value = Mock(tools=[tool])

    wrapper = MCPToolWrapper(session=session, use_openai_restrictions=True)
    tools = await wrapper.get_tools()

    # Name should be sanitized
    assert "/" not in tools[0].name
    assert ":" not in tools[0].name
    assert "@" not in tools[0].name
    assert "_" in tools[0].name


@pytest.mark.asyncio
async def test_openai_name_truncation():
    """Test names > 64 chars are shortened correctly."""
    session = AsyncMock()

    tool = Mock()
    tool.name = "a" * 100  # 100 characters
    tool.description = "Test tool"
    tool.inputSchema = {"type": "object", "properties": {}}

    session.list_tools.return_value = Mock(tools=[tool])

    wrapper = MCPToolWrapper(session=session, use_openai_restrictions=True)
    tools = await wrapper.get_tools()

    # Name should be truncated to 64 chars
    assert len(tools[0].name) <= 64


@pytest.mark.asyncio
async def test_openai_name_prefix_removal():
    """Test get_, _by_, _with_ removed when needed."""
    session = AsyncMock()

    tool = Mock()
    tool.name = "get_data_by_id_with_filter_and_sort_operations_for_database_query"
    tool.description = "Test tool"
    tool.inputSchema = {"type": "object", "properties": {}}

    session.list_tools.return_value = Mock(tools=[tool])

    wrapper = MCPToolWrapper(session=session, use_openai_restrictions=True)
    tools = await wrapper.get_tools()

    # Verify name is shortened and common patterns removed
    assert len(tools[0].name) <= 64
    # Check that at least one pattern was removed
    original_would_be_too_long = len(tool.name) > 64
    if original_would_be_too_long:
        assert "get_" not in tools[0].name or "_by_" not in tools[0].name


# ============================================================================
# Profiling Tests
# ============================================================================

@pytest.mark.asyncio
async def test_profiling_callback_invoked(mock_mcp_session):
    """Test profile_callback is called with correct data."""
    callback_data = []

    def profile_callback(data):
        callback_data.append(data)

    wrapper = MCPToolWrapper(
        session=mock_mcp_session,
        enable_profiling=True,
        profile_callback=profile_callback,
        current_query_index=5
    )

    tools = await wrapper.get_tools()
    tool = tools[0]

    # Execute tool
    await tool.coroutine(
        data={"id": 1},
        key_name="id",
        value="1",
        condition="equal_to"
    )

    # Verify callback was invoked
    assert len(callback_data) == 1

    # Verify callback data structure
    data = callback_data[0]
    assert "tool_name" in data
    assert "duration_s" in data
    assert "query_index" in data

    assert data["tool_name"] == "filter_data"
    assert isinstance(data["duration_s"], float)
    assert data["duration_s"] >= 0
    assert data["query_index"] == 5


@pytest.mark.asyncio
async def test_profiling_timing_accuracy(mock_mcp_session):
    """Test duration measurements are reasonable."""
    callback_data = []

    def profile_callback(data):
        callback_data.append(data)

    wrapper = MCPToolWrapper(
        session=mock_mcp_session,
        enable_profiling=True,
        profile_callback=profile_callback
    )

    tools = await wrapper.get_tools()
    tool = tools[0]

    # Execute tool
    await tool.coroutine(
        data={"id": 1},
        key_name="id",
        value="1",
        condition="equal_to"
    )

    # Verify timing is reasonable (should be very fast for mock)
    assert callback_data[0]["duration_s"] < 1.0  # Should be < 1 second


@pytest.mark.asyncio
async def test_profiling_disabled_by_default(mock_mcp_session):
    """Test no profiling overhead when disabled."""
    callback_data = []

    def profile_callback(data):
        callback_data.append(data)

    # Profiling disabled
    wrapper = MCPToolWrapper(
        session=mock_mcp_session,
        enable_profiling=False,
        profile_callback=profile_callback
    )

    tools = await wrapper.get_tools()
    tool = tools[0]

    # Execute tool
    await tool.coroutine(
        data={"id": 1},
        key_name="id",
        value="1",
        condition="equal_to"
    )

    # Callback should not be invoked
    assert len(callback_data) == 0


# ============================================================================
# Edge Cases
# ============================================================================

@pytest.mark.asyncio
async def test_tool_without_schema(mock_mcp_session):
    """Test tool without inputSchema still works."""
    tool = Mock()
    tool.name = "no_schema_tool"
    tool.description = "Tool without schema"
    # No inputSchema attribute

    mock_mcp_session.list_tools.return_value = Mock(tools=[tool])

    wrapper = MCPToolWrapper(session=mock_mcp_session)
    tools = await wrapper.get_tools()

    # Should still create tool, but with None args_schema
    assert len(tools) == 1
    assert tools[0].args_schema is None


@pytest.mark.asyncio
async def test_tool_with_empty_schema(mock_mcp_session):
    """Test tool with empty inputSchema."""
    tool = Mock()
    tool.name = "empty_schema_tool"
    tool.description = "Tool with empty schema"
    tool.inputSchema = {"type": "object"}  # No properties

    mock_mcp_session.list_tools.return_value = Mock(tools=[tool])

    wrapper = MCPToolWrapper(session=mock_mcp_session)
    tools = await wrapper.get_tools()

    # Should create tool with None args_schema (no fields)
    assert len(tools) == 1
    assert tools[0].args_schema is None


@pytest.mark.asyncio
async def test_tool_result_without_content(mock_mcp_session):
    """Test tool result without content."""
    # Mock call_tool to return result without content
    mock_mcp_session.call_tool.return_value = Mock(content=None)

    wrapper = MCPToolWrapper(session=mock_mcp_session)
    tools = await wrapper.get_tools()

    tool = tools[0]
    result = await tool.coroutine(
        data={"id": 1},
        key_name="id",
        value="1",
        condition="equal_to"
    )

    # Should return "No result"
    assert result == "No result"
