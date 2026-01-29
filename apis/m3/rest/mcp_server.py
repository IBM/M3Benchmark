"""
MCP Server Wrapper for FastAPI Server
Dynamically discovers FastAPI endpoints and exposes them as MCP tools
"""
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import httpx
from mcp.server import Server
from mcp.types import Tool, TextContent
from pydantic import AnyUrl

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FastAPIMCPServer:
    """MCP Server that wraps a FastAPI server dynamically"""

    def __init__(self, fastapi_base_url: str = "http://localhost:8000"):
        self.fastapi_base_url = fastapi_base_url
        self.server = Server("fastapi-mcp-wrapper")
        self.tools_cache: List[Tool] = []
        self.openapi_spec: Optional[Dict] = None

        # Register handlers
        self.server.list_tools()(self.list_tools)
        self.server.call_tool()(self.call_tool)

    async def discover_fastapi_endpoints(self) -> Dict:
        """Fetch OpenAPI spec from FastAPI server"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.fastapi_base_url}/openapi.json")
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch OpenAPI spec: {e}")
            raise

    def convert_openapi_to_mcp_tools(self, openapi_spec: Dict) -> List[Tool]:
        """Convert OpenAPI spec to MCP tools"""
        tools = []
        paths = openapi_spec.get("paths", {})

        for path, path_item in paths.items():
            for method, operation in path_item.items():
                if method.lower() not in ["get", "post", "put", "delete", "patch"]:
                    continue

                # Create tool name from operationId or path
                operation_id = operation.get("operationId")
                if operation_id:
                    tool_name = operation_id
                else:
                    # Generate name from path
                    tool_name = path.replace("/", "_").replace("{", "").replace("}", "").strip("_")

                # Get description
                description = operation.get("summary", operation.get("description", f"{method.upper()} {path}"))

                # Build input schema from parameters
                input_schema = {
                    "type": "object",
                    "properties": {},
                    "required": []
                }

                # Path parameters
                path_params = [p for p in operation.get("parameters", []) if p.get("in") == "path"]
                # Query parameters
                query_params = [p for p in operation.get("parameters", []) if p.get("in") == "query"]

                for param in path_params + query_params:
                    param_name = param["name"]
                    param_schema = param.get("schema", {})

                    input_schema["properties"][param_name] = {
                        "type": param_schema.get("type", "string"),
                        "description": param.get("description", "")
                    }

                    if param.get("required", False):
                        input_schema["required"].append(param_name)

                # Request body for POST/PUT/PATCH
                if method.lower() in ["post", "put", "patch"]:
                    request_body = operation.get("requestBody", {})
                    if request_body:
                        content = request_body.get("content", {})
                        json_schema = content.get("application/json", {}).get("schema", {})
                        if json_schema:
                            input_schema["properties"]["body"] = json_schema
                            if request_body.get("required", False):
                                input_schema["required"].append("body")

                tool = Tool(
                    name=tool_name,
                    description=description,
                    inputSchema=input_schema
                )

                # Store metadata for calling the tool
                tool._metadata = {
                    "path": path,
                    "method": method.upper(),
                    "path_params": [p["name"] for p in path_params],
                    "query_params": [p["name"] for p in query_params]
                }

                tools.append(tool)

        return tools

    async def initialize(self):
        """Initialize the server by discovering FastAPI endpoints"""
        logger.info(f"Discovering FastAPI endpoints from {self.fastapi_base_url}")
        self.openapi_spec = await self.discover_fastapi_endpoints()
        self.tools_cache = self.convert_openapi_to_mcp_tools(self.openapi_spec)
        logger.info(f"Discovered {len(self.tools_cache)} tools")

    async def list_tools(self) -> List[Tool]:
        """List all available tools"""
        if not self.tools_cache:
            await self.initialize()
        return self.tools_cache

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Call a FastAPI endpoint as an MCP tool"""
        # Find the tool
        tool = None
        for t in self.tools_cache:
            if t.name == name:
                tool = t
                break

        if not tool:
            raise ValueError(f"Tool {name} not found")

        metadata = tool._metadata
        path = metadata["path"]
        method = metadata["method"]
        path_params = metadata["path_params"]
        query_params = metadata["query_params"]

        # Build the URL
        url = self.fastapi_base_url + path

        # Replace path parameters
        for param_name in path_params:
            if param_name in arguments:
                url = url.replace(f"{{{param_name}}}", str(arguments[param_name]))

        # Build query parameters
        query_dict = {}
        for param_name in query_params:
            if param_name in arguments:
                query_dict[param_name] = arguments[param_name]

        if query_dict:
            url = f"{url}?{urlencode(query_dict)}"

        # Get request body if present
        body = arguments.get("body")

        # Make the HTTP request
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                if method == "GET":
                    response = await client.get(url)
                elif method == "POST":
                    response = await client.post(url, json=body)
                elif method == "PUT":
                    response = await client.put(url, json=body)
                elif method == "DELETE":
                    response = await client.delete(url)
                elif method == "PATCH":
                    response = await client.patch(url, json=body)
                else:
                    raise ValueError(f"Unsupported method: {method}")

                response.raise_for_status()
                result = response.json()

                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )]
        except httpx.HTTPError as e:
            logger.error(f"HTTP error calling {method} {url}: {e}")
            return [TextContent(
                type="text",
                text=f"Error calling API: {str(e)}"
            )]
        except Exception as e:
            logger.error(f"Error calling {method} {url}: {e}")
            return [TextContent(
                type="text",
                text=f"Error: {str(e)}"
            )]

    async def run(self):
        """Run the MCP server"""
        from mcp.server.stdio import stdio_server

        # Initialize before running
        await self.initialize()

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


async def main():
    """Main entry point"""
    import sys

    # Get FastAPI URL from environment or use default
    import os
    fastapi_url = os.getenv("FASTAPI_BASE_URL", "http://localhost:8000")

    logger.info(f"Starting MCP server wrapping FastAPI at {fastapi_url}")

    server = FastAPIMCPServer(fastapi_base_url=fastapi_url)
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
