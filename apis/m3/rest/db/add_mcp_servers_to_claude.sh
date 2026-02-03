#!/bin/bash
# Adds MCP servers to Claude Code for each domain in domains.json
# Then checks each server's tools for overly long names.
# USAGE: bash /Users/anu/Documents/GitHub/routing/EnterpriseBenchmark/apis/m3/rest/db/add_mcp_servers.sh
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CLAUDE_BIN="/Users/anu/Library/Application Support/Claude/claude-code/2.1.5/claude"
DOMAINS_FILE="$SCRIPT_DIR/domains.json"
MAX_TOOL_NAME_LENGTH=64

for domain in $(jq -r '.[]' "$DOMAINS_FILE"); do
  echo "===== Adding MCP server for: $domain ====="
  "$CLAUDE_BIN" mcp add "fastapi-mcp-${domain}" -- docker exec -i -e "MCP_DOMAINS=${domain}" fastapi-mcp-server python mcp_server.py

  # Check tool names by sending a tools/list request to the server
  echo "  Checking tools for $domain..."
  INIT_REQUEST='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"check","version":"0.1"}}}'
  INITIALIZED_NOTIF='{"jsonrpc":"2.0","method":"notifications/initialized"}'
  TOOLS_REQUEST='{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'

  RESPONSE=$(printf '%s\n%s\n%s\n' "$INIT_REQUEST" "$INITIALIZED_NOTIF" "$TOOLS_REQUEST" \
    | docker exec -i -e "MCP_DOMAINS=${domain}" fastapi-mcp-server python mcp_server.py 2>/dev/null)

  # Parse the tools/list response (the line containing "id":2)
  TOOLS_LINE=$(echo "$RESPONSE" | grep '"id":2' | head -1)

  if [ -n "$TOOLS_LINE" ]; then
    LONG_TOOLS=$(echo "$TOOLS_LINE" | jq -r '.result.tools[]?.name // empty' 2>/dev/null \
      | awk -v max="$MAX_TOOL_NAME_LENGTH" '{ if (length > max) print "    WARNING: tool name too long (" length " chars): " $0 }')

    TOOL_COUNT=$(echo "$TOOLS_LINE" | jq '.result.tools | length' 2>/dev/null)
    echo "  Found $TOOL_COUNT tools for $domain"

    if [ -n "$LONG_TOOLS" ]; then
      echo "$LONG_TOOLS"
    else
      echo "  All tool names are <= $MAX_TOOL_NAME_LENGTH chars"
    fi
  else
    echo "  Could not retrieve tools list for $domain"
  fi

  echo ""
done

echo "Done."
