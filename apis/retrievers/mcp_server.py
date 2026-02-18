"""
MCP Server Wrapper for ChromaDB Retriever FastAPI Server.
Dynamically discovers FastAPI endpoints and exposes them as MCP tools.

Supports filtering by domain name (e.g., MCP_DOMAIN="address" only exposes
the POST /{domain}/query endpoint for that domain).

Usage:
    # Start the FastAPI retriever server first:
    cd apis/retrievers && python run.py

    # Then in another terminal, run the MCP server:
    MCP_DOMAIN=address python mcp_server.py

Environment variables:
    FASTAPI_BASE_URL: Base URL of the FastAPI server (default: http://localhost:8001)
    MCP_SERVER_NAME:  Name for this MCP server instance (default: retriever-mcp)
    MCP_DOMAIN:       Domain to expose (e.g., "address", "hockey").
                      Maps to /{domain}/query endpoint.
                      If not set, all /{domain}/query endpoints are exposed.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional
import httpx
from mcp.server import Server
from mcp.types import Tool, TextContent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FastAPIMCPServer:
    """MCP Server that wraps the ChromaDB Retriever FastAPI server.

    Args:
        fastapi_base_url: Base URL of the FastAPI server.
        server_name: Name for this MCP server instance.
        domains: List of domain names to include (e.g., ["address", "hockey"]).
                 Each domain maps to /{domain}/* path prefix.
                 If None, all endpoints are exposed.
    """

    def __init__(
        self,
        fastapi_base_url: str = "http://localhost:8001",
        server_name: str = "retriever-mcp",
        domains: Optional[List[str]] = None,
    ):
        self.fastapi_base_url = fastapi_base_url
        self.server = Server(server_name)
        self.tools_cache: List[Tool] = []
        self.openapi_spec: Optional[Dict] = None
        self.domains = domains

        # Register handlers
        self.server.list_tools()(self.list_tools)
        self.server.call_tool()(self.call_tool)

    async def discover_fastapi_endpoints(self) -> Dict:
        """Fetch OpenAPI spec from FastAPI server."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.fastapi_base_url}/openapi.json")
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch OpenAPI spec: {e}")
            raise

    def _is_query_endpoint(self, path: str, method: str) -> bool:
        """Check if an endpoint is a POST query endpoint (e.g., /{domain}/query)."""
        return method.lower() == "post" and path.endswith("/query")

    def _resolve_refs(self, schema: Any, spec: Dict) -> Any:
        """Recursively resolve $ref pointers in a JSON Schema using the OpenAPI spec."""
        if isinstance(schema, dict):
            if "$ref" in schema:
                ref_path = schema["$ref"]  # e.g. "#/components/schemas/QueryRequest"
                if ref_path.startswith("#/"):
                    parts = ref_path[2:].split("/")
                    resolved = spec
                    for part in parts:
                        resolved = resolved.get(part, {})
                    return self._resolve_refs(resolved, spec)
                return schema  # external ref, leave as-is
            return {k: self._resolve_refs(v, spec) for k, v in schema.items()}
        if isinstance(schema, list):
            return [self._resolve_refs(item, spec) for item in schema]
        return schema

    def _extract_path_domain(self, path: str) -> Optional[str]:
        """Extract domain from a concrete path like /address/query -> address."""
        parts = path.strip("/").split("/")
        return parts[0] if len(parts) >= 2 else None

    def convert_openapi_to_mcp_tools(self, openapi_spec: Dict) -> List[Tool]:
        """Convert OpenAPI spec to MCP tools.

        Each concrete path like /address/query becomes one tool (query_address).
        When domains are specified, only matching paths are included.
        """
        tools = []
        paths = openapi_spec.get("paths", {})

        for path, path_item in paths.items():
            for method, operation in path_item.items():
                if method.lower() not in ["get", "post", "put", "delete", "patch"]:
                    continue

                if not self._is_query_endpoint(path, method):
                    continue

                # Extract domain from concrete path (e.g., /address/query -> address)
                path_domain = self._extract_path_domain(path)
                if not path_domain:
                    continue

                # Filter to specified domains
                if self.domains and path_domain not in self.domains:
                    continue

                # Get request body schema (resolve $ref pointers)
                body_schema = None
                if method.lower() in ["post", "put", "patch"]:
                    request_body = operation.get("requestBody", {})
                    if request_body:
                        content = request_body.get("content", {})
                        raw_schema = content.get("application/json", {}).get("schema", {})
                        body_schema = self._resolve_refs(raw_schema, openapi_spec)

                # Build input schema — flatten body props to top level
                input_schema = {
                    "type": "object",
                    "properties": {},
                    "required": [],
                }
                body_param_names = []
                if body_schema:
                    body_props = body_schema.get("properties", {})
                    body_req = body_schema.get("required", [])
                    for prop_name, prop_schema in body_props.items():
                        input_schema["properties"][prop_name] = prop_schema
                        body_param_names.append(prop_name)
                        if prop_name in body_req:
                            input_schema["required"].append(prop_name)

                tool_name = f"query_{path_domain}"
                description = operation.get("description") or f"Semantic search within the '{path_domain}' domain collection. Pass a natural language question and get relevant document chunks back."

                tool = Tool(
                    name=tool_name,
                    description=description,
                    inputSchema=input_schema,
                )

                # Store metadata for calling the tool
                tool._metadata = {
                    "path": path,
                    "method": method.upper(),
                    "body_params": body_param_names,
                }

                tools.append(tool)

        return tools

    async def initialize(self):
        """Initialize the server by discovering FastAPI endpoints."""
        logger.info(f"Discovering FastAPI endpoints from {self.fastapi_base_url}")
        if self.domains:
            logger.info(f"Filtering to domains: {self.domains}")

        self.openapi_spec = await self.discover_fastapi_endpoints()
        self.tools_cache = self.convert_openapi_to_mcp_tools(self.openapi_spec)
        logger.info(f"Discovered {len(self.tools_cache)} tools")

    async def list_tools(self) -> List[Tool]:
        """List all available tools."""
        if not self.tools_cache:
            await self.initialize()
        return self.tools_cache

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Call a FastAPI endpoint as an MCP tool."""
        # Find the tool
        tool = None
        for t in self.tools_cache:
            if t.name == name:
                tool = t
                break

        if not tool:
            raise ValueError(f"Tool {name} not found")

        metadata = tool._metadata
        url = self.fastapi_base_url + metadata["path"]
        method = metadata["method"]

        # Reconstruct request body from flattened arguments
        body_params = metadata.get("body_params", [])
        body = {k: arguments[k] for k in body_params if k in arguments} or None

        # Make the HTTP request
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=body)
                response.raise_for_status()
                result = response.json()

                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2),
                )]
        except httpx.HTTPError as e:
            logger.error(f"HTTP error calling {method} {url}: {e}")
            return [TextContent(
                type="text",
                text=f"Error calling API: {str(e)}",
            )]
        except Exception as e:
            logger.error(f"Error calling {method} {url}: {e}")
            return [TextContent(
                type="text",
                text=f"Error: {str(e)}",
            )]

    async def run(self):
        """Run the MCP server over stdio."""
        from mcp.server.stdio import stdio_server

        # Initialize before running
        await self.initialize()

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )


def parse_list_env(env_var: str) -> Optional[List[str]]:
    """Parse comma-separated environment variable into list."""
    value = os.getenv(env_var, "")
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


async def main():
    """Main entry point."""
    fastapi_url = os.getenv("FASTAPI_BASE_URL", "http://localhost:8001")
    server_name = os.getenv("MCP_SERVER_NAME", "retriever-mcp")
    domains = parse_list_env("MCP_DOMAIN")

    logger.info(f"Starting MCP server '{server_name}' wrapping FastAPI at {fastapi_url}")

    server = FastAPIMCPServer(
        fastapi_base_url=fastapi_url,
        server_name=server_name,
        domains=domains,
    )
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
