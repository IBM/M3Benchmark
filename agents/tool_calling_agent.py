import json
from typing import Any
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from agents.result_handle_manager import ResultHandleManager

class ToolCallingAgent:
    """Agent that executes tool-calling loops with handle-based result management"""

    def __init__(
        self,
        llm_with_tools: Any,
        mcp_tools: list,
        initial_data_handle: str,
        max_iterations: int = 10
    ):
        """Initialize agent with LLM and tools via dependency injection

        Args:
            llm_with_tools: LLM instance with tools already bound via llm.bind_tools()
            mcp_tools: List of MCP tool instances
            initial_data_handle: Handle string for initial dataset
            max_iterations: Maximum number of agent loop iterations (default: 10)
        """
        self.handle_manager = ResultHandleManager()
        self._llm_with_tools = llm_with_tools
        self._mcp_tools = mcp_tools
        self._initial_data_handle = initial_data_handle
        self._max_iterations = max_iterations
        self._messages = []

    def _build_system_message(self) -> str:
        """Build system message explaining handle system"""
        # Get initial data reference with key_names and preview
        initial_data_info = ""
        try:
            initial_data = self.handle_manager.get_result(self._initial_data_handle)
            initial_data_reference = self.handle_manager.create_reference(
                self._initial_data_handle,
                initial_data
            )
            # Parse the reference to extract key_names and preview
            ref_dict = json.loads(initial_data_reference)
            if ref_dict.get("type") == "data_table":
                key_names = ref_dict.get("key_names", [])
                preview = ref_dict.get("preview", {})
                initial_data_info = f"""
- Available columns (key_names): {key_names}
- Preview of first few rows: {json.dumps(preview, indent=2)}"""
        except (ValueError, KeyError, json.JSONDecodeError):
            # If we can't get the data, just show the handle
            pass

        return f"""
You are a helpful data analysis assistant. Use the available tools and data to answer queries.

IMPORTANT - Tool Result Handles:
- Tool results are returned as handles/references, not full data
- Data tables have: {{"handle": "filtered_data_1", "type": "data_table", "key_names": [...], "preview": {{...}}}}
- Remember that the "preview" is only the first few values of the full data_table.
- To use a result in another tool, pass the handle string as the 'data' argument
- Example: After filter_data returns {{"handle": "filtered_superhero_1", ...}}
  Call: sort_data(data="filtered_superhero_1", key_name="age", ...)
- Scalar results include both handle and value: {{"handle": "count_1", "type": "scalar", "value": 42}}
- Always use handles to reference previous results in tool chains
- Make at most a single tool call per iteration. Once you have the information necessary to answer the query, return the answer with no additional tool calls.

INITIAL DATA:
- The initial dataset for this task is available as handle: "{self._initial_data_handle}"{initial_data_info}
- Start by using this handle in your first tool call
        """

    async def run(self, query: str) -> str:
        """Execute agent loop for a query and return final answer

        Args:
            query: The natural language query to process

        Returns:
            Final answer as string, or indication of max iterations reached
        """
        # Initialize message history
        self._messages = [
            SystemMessage(content=self._build_system_message()),
            HumanMessage(content=query)
        ]

        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}\n")

        print("SYSTEM:")
        print(self._build_system_message())

        # Run agent loop
        for iteration in range(self._max_iterations):
            print(f"\n{'='*80}")
            print(f"ITERATION {iteration + 1}/{self._max_iterations}")
            print(f"{'='*80}")
            print(f"Message history length: {len(self._messages)}")

            # Invoke LLM
            print("Invoking LLM...")
            response = await self._llm_with_tools.ainvoke(self._messages)
            print("LLM response received")

            if response.content:
                content_preview = str(response.content)[:200]
                print(f"Response content (first 200 chars): {content_preview}")

            # Add AI response to messages
            self._messages.append(response)

            # Check if done (no tool calls)
            if not response.tool_calls:
                print("\nNo tool calls - Agent is done!")
                print(f"Final Answer: {response.content}")
                return response.content

            # Execute tool calls
            print(f"\nNumber of tool calls: {len(response.tool_calls)}")
            for idx, tool_call in enumerate(response.tool_calls, 1):
                print(f"\n--- Tool Call {idx}/{len(response.tool_calls)} ---")
                await self._execute_tool_call(tool_call)

        # Max iterations reached
        print(f"\nReached max iterations ({self._max_iterations})")
        return f"Max iterations reached ({self._max_iterations})"

    async def _execute_tool_call(self, tool_call: dict):
        """Execute a single tool call and update message history

        Args:
            tool_call: Dict with 'name', 'args', and 'id' keys
        """
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        tool_id = tool_call['id']

        print("\n  === Executing Tool Call ===")
        print(f"  Tool: {tool_name}")
        print(f"  Tool ID: {tool_id}")

        # Unwrap kwargs if present (fixes LLM schema interpretation issue)
        if len(tool_args) == 1 and 'kwargs' in tool_args:
            print("  Unwrapping kwargs...")
            tool_args = tool_args['kwargs']

        print("  Original Arguments:")
        for k, v in tool_args.items():
            print(f"    {k}: {v}")

        # Find matching tool
        matching_tool = next((t for t in self._mcp_tools if t.name == tool_name), None)

        if not matching_tool:
            print(f"  ERROR: Tool '{tool_name}' not found in available tools!")
            error_msg = json.dumps({"error": f"Tool {tool_name} not found"})
            self._messages.append(ToolMessage(
                content=error_msg,
                tool_call_id=tool_id,
                name=tool_name
            ))
            return

        try:
            # Resolve any handle references in arguments
            print("  Resolving handle references...")
            resolved_args = self.handle_manager.resolve_args(tool_args, resolve_data=True)
            # print("  Resolved Arguments:")
            # for k, v in resolved_args.items():
            #     v_str = str(v)
            #     if len(v_str) > 200:
            #         v_str = v_str[:200] + "..."
            #     print(f"    {k}: {v_str}")

            # Execute tool with resolved arguments
            print("  Invoking tool...")
            result = await matching_tool.ainvoke(resolved_args)
            # print(f"  Raw result length: {len(result) if result else 0}")
            # print(f"  Raw result (first 500 chars): {str(result)[:500]}")

            parsed_result = json.loads(result)[0]
            print(f"  Parsed result type: {type(parsed_result)}")

            # Extract text field from MCP TextContent format
            if isinstance(parsed_result, dict) and 'type' in parsed_result and 'text' in parsed_result:
                print("  Extracting text from MCP TextContent format...")
                parsed_result = parsed_result['text']

                # If the text field contains JSON, parse it
                if isinstance(parsed_result, str):
                    try:
                        parsed_result = json.loads(parsed_result)
                        print("  Successfully parsed JSON from text field")
                    except (json.JSONDecodeError, ValueError):
                        # Not JSON, keep as string
                        print("  Text field is not JSON, keeping as string")
                        pass

            print(f"  Final parsed result type: {type(parsed_result)}")
            result_str = str(parsed_result)
            if len(result_str) > 300:
                result_str = result_str[:300] + "..."
            print(f"  Final parsed result: {result_str}")

            # Check if result contains an error
            if isinstance(parsed_result, dict) and "error" in parsed_result:
                # Don't store errors - return them directly in the message
                print(f"  ERROR DETECTED in result: {parsed_result['error']}")
                error_msg = json.dumps(parsed_result)
                self._messages.append(ToolMessage(
                    content=error_msg,
                    tool_call_id=tool_id,
                    name=tool_name
                ))
                print("  Error message sent to LLM")
                return
            elif isinstance(parsed_result, str) and "error" in parsed_result.lower():
                # Handle plain string error messages
                print(f"  ERROR DETECTED in string result: {parsed_result}")
                error_msg = json.dumps({"error": parsed_result})
                self._messages.append(ToolMessage(
                    content=error_msg,
                    tool_call_id=tool_id,
                    name=tool_name
                ))
                print("  Error message sent to LLM")
                return

            # Store parsed result and generate handle
            print("  Storing result in handle manager...")
            handle = self.handle_manager.store_result(tool_name, parsed_result)
            print(f"  Result stored with handle: {handle}")

            # Create reference for message
            print("  Creating reference for LLM...")
            reference = self.handle_manager.create_reference(handle, parsed_result)
            print(f"  Reference created (length: {len(reference)})")
            # print(f"  Reference content (first 300 chars): {reference[:300]}")

            # Add tool result with reference instead of full data
            self._messages.append(ToolMessage(
                content=reference,
                tool_call_id=tool_id,
                name=tool_name
            ))
            print("  SUCCESS: Tool execution complete\n")

        except Exception as e:
            print(f"  EXCEPTION CAUGHT: {type(e).__name__}")
            print(f"  Exception message: {str(e)}")
            import traceback
            print("  Traceback:")
            traceback.print_exc()
            error_msg = json.dumps({"error": str(e)})
            self._messages.append(ToolMessage(
                content=error_msg,
                tool_call_id=tool_id,
                name=tool_name
            ))
            print("  Error message sent to LLM\n")

    def restart(self):
        """Clear agent state for new query execution"""
        self.handle_manager.clear()
        self._messages = []
