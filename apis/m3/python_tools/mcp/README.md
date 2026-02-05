# Slot-Filling MCP Server

An MCP (Model Context Protocol) server that exposes slot-filling data manipulation tools for AI agents.

## Installation

```bash
pip install invocable-api-hub
```

## Quick Start

```bash
# Run with a database
slot-filling-mcp --db /path/to/your/database.sqlite

# Run with specific tables
slot-filling-mcp --db /path/to/database.sqlite --tables users,orders,products

# Run with config file
slot-filling-mcp --config ./slot-filling-mcp.json

# Enable verbose logging
slot-filling-mcp --db /path/to/database.sqlite --verbose
```

## Configuration

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--db PATH` | Path to SQLite database file |
| `--tables TABLE1,TABLE2` | Comma-separated list of tables to load (default: all) |
| `--config FILE` | Path to JSON config file |
| `--cache-dir DIR` | Directory for caching (default: ./mcp_cache) |
| `-v, --verbose` | Enable verbose logging |
| `--version` | Show version |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `SLOT_FILLING_MCP_DATABASE` | Path to SQLite database |
| `SLOT_FILLING_MCP_CACHE_DIR` | Cache directory |
| `SLOT_FILLING_MCP_TABLES` | Comma-separated table list |

### Config File

Create `slot-filling-mcp.json` in your working directory or `~/.config/slot-filling-mcp/config.json`:

```json
{
    "database_path": "/path/to/database.sqlite",
    "tables": ["users", "orders"],
    "cache_dir": "./cache"
}
```

Configuration priority (highest to lowest):
1. Command line arguments
2. Environment variables
3. Config file
4. Default values

## Available Tools

### Data Manipulation Tools

| Tool | Description |
|------|-------------|
| `filter_data` | Filter data based on conditions (equal_to, greater_than, like, etc.) |
| `sort_data` | Sort data by a key (ascending/descending) |
| `aggregate_data` | Perform aggregations (min, max, sum, count, mean, std) |
| `transform_data` | Apply transformations (substring, abs, datetime) |
| `retrieve_data` | Select specific columns from data |
| `concatenate_data` | Combine two datasets |
| `select_unique_values` | Get unique values from an array |
| `peek_fcn` | Inspect data structure and content |

### Helper Tools

| Tool | Description |
|------|-------------|
| `list_tables` | List all preloaded tables with their columns and row counts |
| `get_table` | Get the full data from a specific table |

## Claude Desktop Integration

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
    "mcpServers": {
        "slot-filling": {
            "command": "slot-filling-mcp",
            "args": ["--config", "/path/to/slot-filling-mcp.json"]
        }
    }
}
```

Or with inline database path:

```json
{
    "mcpServers": {
        "slot-filling": {
            "command": "slot-filling-mcp",
            "args": ["--db", "/path/to/your/database.sqlite"]
        }
    }
}
```

## Python Module Usage

Run as a module:

```bash
python -m apis.m3.python_tools.mcp --db /path/to/db.sqlite
```

Or use programmatically:

```python
import asyncio
from apis.m3.python_tools.mcp import (
    create_server,
    MCPServerConfig,
)

config = MCPServerConfig(
    database_path="/path/to/database.sqlite",
    tables=["users", "orders"],
)

server = create_server(config)
asyncio.run(server.run())
```

## Example Workflow

Once connected to an AI agent (like Claude), you can:

1. **List available tables:**
   ```
   Use the list_tables tool to see what data is available.
   ```

2. **Get table data:**
   ```
   Use get_table with table_name="users" to retrieve the users table.
   ```

3. **Filter data:**
   ```
   Use filter_data with the users data, filtering where users_status equals "active".
   ```

4. **Aggregate results:**
   ```
   Use aggregate_data to count the filtered results.
   ```

## Column Naming Convention

When tables are loaded, column names are prefixed with the table name to avoid collisions:
- Table `users` with column `id` becomes `users_id`
- Table `orders` with column `id` becomes `orders_id`

This allows working with multiple tables without ambiguity.
