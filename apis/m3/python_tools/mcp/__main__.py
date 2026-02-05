"""
Allow running the MCP server as a module:
    python -m apis.m3.python_tools.mcp --db /path/to/db.sqlite
"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
