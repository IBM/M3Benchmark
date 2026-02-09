import json
from typing import Any


class ResultHandleManager:
    """Manages storage and resolution of tool result handles"""

    def __init__(self):
        self.storage: dict[str, Any] = {}
        self.counter: dict[str, int] = {}

    def store_result(self, tool_name: str, result: Any) -> str:
        """Store result and return descriptive handle"""
        handle = self._generate_handle(tool_name, result)
        self.storage[handle] = result
        return handle

    def get_result(self, handle: str) -> Any:
        """Retrieve result by handle"""
        if handle not in self.storage:
            raise ValueError(f"Unknown handle: {handle}")

        result = self.storage[handle]
        return result

    def _generate_handle(self, tool_name: str, result: Any) -> str:
        """Generate descriptive handle like 'filtered_superhero_1'"""
        # Map tool names to operation prefixes
        operation_map = {
            "filter_data": "filtered",
            "sort_data": "sorted",
            "retrieve_data": "retrieved",
            "transform_data": "transformed",
            "aggregate_data": "aggregated",
            "concatenate_data": "concatenated",
            "Calculator": "calculated",
            "select_unique_values": "unique",
            "get_data": "initial"
        }

        prefix = operation_map.get(tool_name, "result")

        # Increment counter for this operation type
        if prefix not in self.counter:
            self.counter[prefix] = 0
        self.counter[prefix] += 1

        # Infer subject from result columns for data tables
        suffix = "data"
        if isinstance(result, dict) and result:
            keys = list(result.keys())
            if keys and '_' in keys[0]:
                suffix = keys[0].split('_')[0]  # e.g., "superhero_id" → "superhero"

        return f"{prefix}_{suffix}_{self.counter[prefix]}"

    def _is_handle(self, value: str) -> bool:
        """Check if string is a valid handle"""
        return value in self.storage

    def resolve_args(self, args: dict, resolve_data: bool = True) -> dict:
        """Replace handle strings in tool args with actual data

        Args:
            args: Tool arguments dict
            resolve_data: If False, keep data handles as strings (for MCP tools)
        """
        resolved = args.copy()

        if resolve_data:
            # Handle 'data' parameter (most tools)
            if 'data' in resolved and isinstance(resolved['data'], str):
                if self._is_handle(resolved['data']):
                    resolved['data'] = self.get_result(resolved['data'])

            # Handle 'data_1', 'data_2' for concatenate_data
            for key in ['data_1', 'data_2']:
                if key in resolved and isinstance(resolved[key], str):
                    if self._is_handle(resolved[key]):
                        resolved[key] = self.get_result(resolved[key])

        return resolved

    def create_reference(self, handle: str, result: Any) -> str:
        """Create compact JSON reference for ToolMessage content"""
        # For scalars (int, float) - include actual value
        if isinstance(result, (int, float)):
            ref = {
                "handle": handle,
                "type": "scalar",
                "value": result
            }
            return json.dumps(ref, indent=2)

        # For data tables (dict)
        if isinstance(result, dict):
            # Result is a plain dict, may contain _dtypes
            dtypes = result.get('_dtypes', None)

            # Get data keys (excluding _dtypes)
            data_keys = [k for k in result.keys() if k != '_dtypes']
            num_records = len(result[data_keys[0]]) if data_keys else 0

            # Create preview (first 3 rows, first 5 columns)
            preview = {}
            for key in data_keys[:5]:
                values = result[key]
                preview[key] = values[:3] if len(values) > 3 else values

            ref = {
                "handle": handle,
                "type": "data_table",
                "num_records": num_records,
                "num_keys": len(data_keys),
                "key_names": data_keys,
                "preview": preview
            }

            # Include _dtypes in reference if present
            if dtypes is not None:
                ref['_dtypes'] = dtypes

            return json.dumps(ref, indent=2)

        # For other types (lists, arrays)
        ref = {
            "handle": handle,
            "type": "array",
            "length": len(result) if hasattr(result, '__len__') else None,
            "preview": str(result)[:200]
        }
        return json.dumps(ref, indent=2)

    def store_initial_data(self, data: Any) -> str:
        """Store initial data and return its handle"""
        return self.store_result("get_data", data)

    def clear(self):
        """Clear all stored results"""
        self.storage.clear()
        self.counter.clear()