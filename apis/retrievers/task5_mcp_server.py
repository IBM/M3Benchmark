"""
Task 5 Combined MCP Server
==========================

Aggregates tools from two FastAPI servers running in the same container and
exposes them as a single MCP server:

  - M3 REST FastAPI   (http://localhost:8000) → SQL/REST tools for the primary domain
  - Retriever FastAPI (http://localhost:8001) → semantic-search tools for the primary
                                                domain AND its negative domains

Domain filtering is asymmetric by design:

  M3 REST tools    — filtered to the primary domain only (MCP_DOMAIN).
                     e.g. MCP_DOMAIN=address → /v1/address/* tools only.

  Retriever tools  — filtered to the primary domain PLUS its "negative" domains
                     from domain_negatives.json. These are confusable domains the
                     agent must distinguish at retrieval time.
                     e.g. MCP_DOMAIN=address →
                       query_address, query_olympics, query_card_games,
                       query_legislator, query_craftbeer

Each tool's ``_metadata`` records which backend URL handles it, so
``call_tool`` can route the HTTP request correctly.

Environment variables:
    MCP_DOMAIN         Primary domain (e.g. "address"). Drives both M3 REST
                       filtering and the negatives lookup. If unset, all
                       endpoints from both servers are exposed.
    M3_REST_BASE_URL   Base URL of the M3 REST FastAPI server
                       (default: http://localhost:8000)
    RETRIEVER_BASE_URL Base URL of the Retriever FastAPI server
                       (default: http://localhost:8001)
    MCP_SERVER_NAME    Name for this MCP server instance
                       (default: task5-combined-mcp)

Usage (inside task_5_m3_environ container):
    docker exec -i -e MCP_DOMAIN=address task_5_m3_environ \\
        python /app/retrievers/task5_mcp_server.py
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import httpx
from mcp.server import Server
from mcp.types import Tool, TextContent

from apis.mcp_logging import _setup_mcp_logging

logger = logging.getLogger(__name__)

# domain_negatives.json lives alongside this script in the container
_NEGATIVES_PATH = Path(__file__).parent / "domain_negatives.json"


def load_retriever_domains(
    primary_domains: Optional[List[str]],
    negatives_path: Path = _NEGATIVES_PATH,
) -> Optional[List[str]]:
    """Return the retriever domains for the given primary domains.

    For each primary domain, the retriever should expose query tools for
    that domain plus its negatives (confusable domains) as listed in
    domain_negatives.json.

    Returns None if primary_domains is None (expose all retriever endpoints).
    """
    if primary_domains is None:
        return None

    try:
        with open(negatives_path) as f:
            negatives_map: Dict[str, List[str]] = json.load(f)
    except FileNotFoundError:
        logger.warning(
            "domain_negatives.json not found at %s — "
            "retriever will use primary domain only.",
            negatives_path,
        )
        return primary_domains

    retriever_domains: List[str] = []
    for domain in primary_domains:
        for d in negatives_map.get(domain, [domain]):
            if d not in retriever_domains:
                retriever_domains.append(d)

    return retriever_domains


class Task5CombinedMCPServer:
    """MCP server that merges M3 REST tools + Retriever tools.

    M3 REST tools are filtered to ``m3_domains`` (the primary domain).
    Retriever tools are filtered to ``retriever_domains`` (primary + negatives).
    """

    def __init__(
        self,
        m3_rest_url: str = "http://localhost:8000",
        retriever_url: str = "http://localhost:8001",
        server_name: str = "task5-combined-mcp",
        m3_domains: Optional[List[str]] = None,
        retriever_domains: Optional[List[str]] = None,
    ):
        self.m3_rest_url = m3_rest_url
        self.retriever_url = retriever_url
        self.m3_domains = m3_domains
        self.retriever_domains = retriever_domains
        self.server = Server(server_name)
        self.tools_cache: List[Tool] = []

        self.server.list_tools()(self.list_tools)
        self.server.call_tool()(self.call_tool)

    # ------------------------------------------------------------------
    # OpenAPI discovery
    # ------------------------------------------------------------------

    async def _fetch_openapi(self, base_url: str) -> Dict:
        """Fetch the OpenAPI JSON spec from a FastAPI server."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/openapi.json")
            response.raise_for_status()
            return response.json()

    # ------------------------------------------------------------------
    # M3 REST tool discovery (mirrors apis/m3/rest/mcp_server.py)
    # Filtered to the primary domain only.
    # ------------------------------------------------------------------

    def _should_include_m3_endpoint(self, path: str) -> bool:
        if not self.m3_domains:
            return True
        return any(path.startswith(f"/v1/{domain}") for domain in self.m3_domains)

    def _m3_rest_tools_from_spec(self, spec: Dict) -> List[Tool]:
        tools = []
        for path, path_item in spec.get("paths", {}).items():
            for method, operation in path_item.items():
                if method.lower() not in ["get", "post", "put", "delete", "patch"]:
                    continue
                if not self._should_include_m3_endpoint(path):
                    continue

                tool_name = operation.get("operationId") or (
                    path.replace("/", "_")
                    .replace("{", "")
                    .replace("}", "")
                    .strip("_")
                )
                description = operation.get(
                    "summary",
                    operation.get("description", f"{method.upper()} {path}"),
                )

                input_schema: Dict[str, Any] = {
                    "type": "object",
                    "properties": {},
                    "required": [],
                }
                path_params = [
                    p for p in operation.get("parameters", []) if p.get("in") == "path"
                ]
                query_params = [
                    p for p in operation.get("parameters", []) if p.get("in") == "query"
                ]
                for param in path_params + query_params:
                    pname = param["name"]
                    pschema = param.get("schema", {})
                    input_schema["properties"][pname] = {
                        "type": pschema.get("type", "string"),
                        "description": param.get("description", ""),
                    }
                    if param.get("required", False):
                        input_schema["required"].append(pname)

                if method.lower() in ["post", "put", "patch"]:
                    req_body = operation.get("requestBody", {})
                    if req_body:
                        content = req_body.get("content", {})
                        json_schema = (
                            content.get("application/json", {}).get("schema", {})
                        )
                        if json_schema:
                            input_schema["properties"]["body"] = json_schema
                            if req_body.get("required", False):
                                input_schema["required"].append("body")

                tool = Tool(
                    name=tool_name,
                    description=description,
                    inputSchema=input_schema,
                )
                tool._metadata = {
                    "backend_url": self.m3_rest_url,
                    "path": path,
                    "method": method.upper(),
                    "path_params": [p["name"] for p in path_params],
                    "query_params": [p["name"] for p in query_params],
                    "body_params": [],
                }
                tools.append(tool)
        return tools

    # ------------------------------------------------------------------
    # Retriever tool discovery (mirrors apis/retrievers/mcp_server.py)
    # Filtered to primary domain + its negatives.
    # ------------------------------------------------------------------

    def _resolve_refs(self, schema: Any, spec: Dict) -> Any:
        """Recursively resolve $ref pointers in a JSON Schema."""
        if isinstance(schema, dict):
            if "$ref" in schema:
                ref_path = schema["$ref"]
                if ref_path.startswith("#/"):
                    parts = ref_path[2:].split("/")
                    resolved = spec
                    for part in parts:
                        resolved = resolved.get(part, {})
                    return self._resolve_refs(resolved, spec)
                return schema
            return {k: self._resolve_refs(v, spec) for k, v in schema.items()}
        if isinstance(schema, list):
            return [self._resolve_refs(item, spec) for item in schema]
        return schema

    def _extract_path_domain(self, path: str) -> Optional[str]:
        parts = path.strip("/").split("/")
        return parts[0] if len(parts) >= 2 else None

    def _retriever_tools_from_spec(self, spec: Dict) -> List[Tool]:
        tools = []
        for path, path_item in spec.get("paths", {}).items():
            for method, operation in path_item.items():
                if method.lower() not in ["get", "post", "put", "delete", "patch"]:
                    continue
                if not (method.lower() == "post" and path.endswith("/query")):
                    continue

                path_domain = self._extract_path_domain(path)
                if not path_domain:
                    continue
                # Use retriever_domains (primary + negatives) for filtering
                if self.retriever_domains and path_domain not in self.retriever_domains:
                    continue

                body_schema = None
                req_body = operation.get("requestBody", {})
                if req_body:
                    content = req_body.get("content", {})
                    raw_schema = content.get("application/json", {}).get("schema", {})
                    body_schema = self._resolve_refs(raw_schema, spec)

                input_schema: Dict[str, Any] = {
                    "type": "object",
                    "properties": {},
                    "required": [],
                }
                body_param_names: List[str] = []
                if body_schema:
                    for prop_name, prop_schema in body_schema.get(
                        "properties", {}
                    ).items():
                        input_schema["properties"][prop_name] = prop_schema
                        body_param_names.append(prop_name)
                        if prop_name in body_schema.get("required", []):
                            input_schema["required"].append(prop_name)

                tool_name = f"query_{path_domain}"
                description = operation.get("description") or (
                    f"Semantic search within the '{path_domain}' domain collection. "
                    "Pass a natural language question and get relevant document chunks back."
                )

                tool = Tool(
                    name=tool_name,
                    description=description,
                    inputSchema=input_schema,
                )
                tool._metadata = {
                    "backend_url": self.retriever_url,
                    "path": path,
                    "method": "POST",
                    "path_params": [],
                    "query_params": [],
                    "body_params": body_param_names,
                }
                tools.append(tool)
        return tools

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    async def initialize(self):
        logger.info(
            "Fetching M3 REST OpenAPI spec from %s (domains: %s)",
            self.m3_rest_url,
            self.m3_domains,
        )
        m3_spec = await self._fetch_openapi(self.m3_rest_url)
        m3_tools = self._m3_rest_tools_from_spec(m3_spec)
        logger.info("M3 REST: %d tools", len(m3_tools))

        logger.info(
            "Fetching Retriever OpenAPI spec from %s (domains: %s)",
            self.retriever_url,
            self.retriever_domains,
        )
        ret_spec = await self._fetch_openapi(self.retriever_url)
        ret_tools = self._retriever_tools_from_spec(ret_spec)
        logger.info("Retriever: %d tools", len(ret_tools))

        self.tools_cache = m3_tools + ret_tools
        logger.info("Total tools exposed: %d", len(self.tools_cache))

    # ------------------------------------------------------------------
    # MCP handlers
    # ------------------------------------------------------------------

    async def list_tools(self) -> List[Tool]:
        if not self.tools_cache:
            await self.initialize()
        return self.tools_cache

    async def call_tool(
        self, name: str, arguments: Dict[str, Any]
    ) -> List[TextContent]:
        logger.info("tool_call tool=%s", name)
        tool = next((t for t in self.tools_cache if t.name == name), None)
        if not tool:
            raise ValueError(f"Tool '{name}' not found")

        meta = tool._metadata
        backend_url: str = meta["backend_url"]
        path: str = meta["path"]
        method: str = meta["method"]
        path_params: List[str] = meta["path_params"]
        query_params: List[str] = meta["query_params"]
        body_params: List[str] = meta.get("body_params", [])

        url = backend_url + path
        for pname in path_params:
            if pname in arguments:
                url = url.replace(f"{{{pname}}}", str(arguments[pname]))

        query_dict = {p: arguments[p] for p in query_params if p in arguments}
        if query_dict:
            url = f"{url}?{urlencode(query_dict)}"

        if body_params:
            body = {k: arguments[k] for k in body_params if k in arguments} or None
        else:
            body = arguments.get("body")

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
                    raise ValueError(f"Unsupported HTTP method: {method}")

                response.raise_for_status()
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(response.json(), indent=2),
                    )
                ]
        except httpx.HTTPError as e:
            logger.error("HTTP error calling %s %s: %s", method, url, e)
            return [TextContent(type="text", text=f"Error calling API: {e}")]
        except Exception as e:
            logger.error("Error calling %s %s: %s", method, url, e)
            return [TextContent(type="text", text=f"Error: {e}")]

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    async def run(self):
        from mcp.server.stdio import stdio_server

        await self.initialize()
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )


def parse_list_env(env_var: str) -> Optional[List[str]]:
    value = os.getenv(env_var, "")
    if not value:
        return None
    return [s.strip() for s in value.split(",") if s.strip()]


async def main():
    _setup_mcp_logging()
    m3_rest_url = os.getenv("M3_REST_BASE_URL", "http://localhost:8000")
    retriever_url = os.getenv("RETRIEVER_BASE_URL", "http://localhost:8001")
    server_name = os.getenv("MCP_SERVER_NAME", "task5-combined-mcp")

    # Primary domain(s) from MCP_DOMAIN (e.g. "address")
    primary_domains = parse_list_env("MCP_DOMAIN")

    # Retriever domains = primary + negatives from domain_negatives.json
    retriever_domains = load_retriever_domains(primary_domains)

    logger.info(
        "Starting Task 5 combined MCP server "
        "(M3 REST @ %s [domains=%s] + Retriever @ %s [domains=%s])",
        m3_rest_url,
        primary_domains,
        retriever_url,
        retriever_domains,
    )

    server = Task5CombinedMCPServer(
        m3_rest_url=m3_rest_url,
        retriever_url=retriever_url,
        server_name=server_name,
        m3_domains=primary_domains,
        retriever_domains=retriever_domains,
    )
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
