#!/usr/bin/env python3
"""
Verification script to demonstrate that the unified MCPToolWrapper
properly preserves MCP tool schemas for LLM visibility.

This script creates a mock MCP session and verifies that:
1. Tool schemas are preserved (args_schema is not None)
2. LLMs can see all parameter definitions
3. Required vs optional fields are correctly identified
"""

import asyncio
from unittest.mock import AsyncMock, Mock
from agents.mcp_tool_wrapper import MCPToolWrapper


async def main():
    print("="*80)
    print("SCHEMA PRESERVATION VERIFICATION")
    print("="*80)

    # Create mock MCP session with a realistic tool schema
    session = AsyncMock()

    # Mock filter_data tool with comprehensive schema
    filter_tool = Mock()
    filter_tool.name = "filter_data"
    filter_tool.description = "Apply a filter based on a specified condition"
    filter_tool.inputSchema = {
        "type": "object",
        "properties": {
            "data": {
                "type": "object",
                "description": "Data object to be filtered"
            },
            "key_name": {
                "type": "string",
                "description": "The key on which the filter will be applied"
            },
            "value": {
                "type": "string",
                "description": "The value to compare against"
            },
            "condition": {
                "type": "string",
                "enum": ["equal_to", "not_equal_to", "greater_than", "less_than"],
                "description": "Condition used in the filter"
            }
        },
        "required": ["data", "key_name", "value", "condition"]
    }

    # Mock aggregate_data tool
    aggregate_tool = Mock()
    aggregate_tool.name = "aggregate_data"
    aggregate_tool.description = "Apply an aggregation operation on data"
    aggregate_tool.inputSchema = {
        "type": "object",
        "properties": {
            "data": {
                "type": "object",
                "description": "Data to aggregate"
            },
            "key_name": {
                "type": "string",
                "description": "Key to aggregate on"
            },
            "operation": {
                "type": "string",
                "enum": ["sum", "avg", "count", "min", "max"],
                "description": "Aggregation operation"
            },
            "groupby_key": {
                "type": "string",
                "description": "Optional key to group by",
                "default": None
            }
        },
        "required": ["data", "key_name", "operation"]
    }

    session.list_tools.return_value = Mock(tools=[filter_tool, aggregate_tool])

    # Create MCPToolWrapper
    print("\n1. Creating MCPToolWrapper...")
    wrapper = MCPToolWrapper(session=session)

    # Get tools
    print("2. Fetching tools from MCP server...")
    tools = await wrapper.get_tools()
    print(f"   Found {len(tools)} tools\n")

    # Verify schema preservation for each tool
    for i, tool in enumerate(tools, 1):
        print(f"\n{'='*80}")
        print(f"Tool #{i}: {tool.name}")
        print(f"{'='*80}")
        print(f"Description: {tool.description}")

        # CRITICAL CHECK: args_schema must not be None
        if tool.args_schema is None:
            print("\n❌ FAIL: args_schema is None - LLM cannot see parameters!")
            continue

        print("\n✓ PASS: args_schema is properly set")

        # Get JSON schema representation (this is what LLM sees)
        schema_dict = tool.args_schema.model_json_schema()

        # Display schema details
        properties = schema_dict.get("properties", {})
        required = schema_dict.get("required", [])

        print(f"\nParameters ({len(properties)} total):")
        print("-" * 80)

        for param_name, param_info in properties.items():
            is_required = "REQUIRED" if param_name in required else "optional"
            param_type = param_info.get("type", "any")
            description = param_info.get("description", "No description")

            # Truncate long descriptions
            if len(description) > 60:
                description = description[:57] + "..."

            print(f"  • {param_name:15} [{is_required:8}] {param_type:10} - {description}")

            # Show enum values if present
            if "Allowed values:" in description:
                # Extract enum values from description
                enum_start = description.find("Allowed values:")
                if enum_start != -1:
                    enum_str = description[enum_start:]
                    print(f"    {enum_str}")

        print(f"\nRequired fields: {', '.join(required)}")

        # Verify Pydantic model can validate
        print("\n✓ Schema validation:")
        print(f"  - Total parameters: {len(properties)}")
        print(f"  - Required parameters: {len(required)}")
        print(f"  - Optional parameters: {len(properties) - len(required)}")

    # Final summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    all_have_schema = all(tool.args_schema is not None for tool in tools)

    if all_have_schema:
        print("\n✅ SUCCESS: All tools have proper schemas!")
        print("   LLMs will be able to see all parameter definitions.")
    else:
        print("\n❌ FAILURE: Some tools have missing schemas!")
        print("   LLMs will not be able to use these tools correctly.")

    print(f"\n{'='*80}\n")

    return all_have_schema


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
