# Tool Checksum Integrity System

Ensures the MCP server returns the exact tools expected for a given
`(capability_id, domain)` pair before any benchmark run proceeds.

---

## How It Works

```
GENERATION (repo maintainer, run once per tool change)
──────────────────────────────────────────────────────

  Docker MCP server
  ┌─────────────────┐
  │  list_tools()   │──► [tool_a, tool_b, tool_c, ...]
  └─────────────────┘
          │
          ▼
  sort by name
  serialize to canonical JSON          ← order-independent, deterministic
  SHA-256 hash
          │
          ▼
  tool_checksums.json
  ┌───────────────────────────────────────┐
  │  "2": {                               │
  │    "california_schools": "40ac55...", │  ← committed to git
  │    "card_games":         "9db121...", │
  │    ...                                │
  │  },                                   │
  │  "3": { ... },                        │
  │  "4": { ... }                         │
  └───────────────────────────────────────┘


VERIFICATION (every benchmark run, opt-in via MCP_VERIFY_CHECKSUMS=1)
──────────────────────────────────────────────────────────────────────

  benchmark_runner.py
       │
       │  MCPToolWrapper(capability_id=2, domain="address")
       ▼
  session.list_tools()  ──►  tools from live MCP server
       │
       ▼
  compute_tool_checksum(live tools)
       │
       ├── matches tool_checksums.json["2"]["address"]?
       │
       ├─── YES ──► proceed with benchmark
       │
       └─── NO  ──► ValueError: checksum mismatch
                    (wrong domain connected, or tools changed unexpectedly)
```

---

## Decision Logic in `verify_checksum()`

```
MCP_VERIFY_CHECKSUMS set?
│
├── NO  ──► skip (no-op, default behaviour)
│
└── YES
      │
      ├── capability not in tool_checksums.json?  ──► WARNING, skip
      │
      ├── domain not in tool_checksums.json?      ──► WARNING, skip
      │
      └── checksum mismatch?
            ├── NO  ──► OK, continue
            └── YES ──► ValueError (hard abort)
```

---

## Files

| File | Role |
|------|------|
| [`tool_checksums.json`](../tool_checksums.json) | Committed reference checksums, one per `(capability, domain)` |
| [`environment/tool_checksums.py`](../environment/tool_checksums.py) | `compute_tool_checksum()`, `verify_checksum()`, `load_checksums()` |
| [`generate_checksums.py`](../generate_checksums.py) | Re-generates `tool_checksums.json` from live Docker servers (maintainers only) |
| [`agents/mcp_tool_wrapper.py`](../agents/mcp_tool_wrapper.py) | Calls `verify_checksum()` inside `get_tools()` when `capability_id` + `domain` are set |

---

## Usage

**Enable verification at runtime:**
```bash
MCP_VERIFY_CHECKSUMS=1 python benchmark_runner.py ...
```

**Regenerate after intentional tool changes:**
```bash
# Requires: docker compose up -d && make download
python generate_checksums.py

# Specific capability / domain only
python generate_checksums.py --capability 2 --domain address

# Preview without writing
python generate_checksums.py --dry-run
```

**Run tests:**
```bash
python -m pytest tests/test_tool_checksums.py -v
```
