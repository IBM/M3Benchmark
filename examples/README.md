# Examples

Scripts for connecting to the benchmark environment, exploring tools, and running your agent against benchmark data.

---

## How the environment works

The benchmark runs four capability containers, each exposing an MCP server. Your agent connects to a container, receives a set of tools, and must answer natural-language questions by calling the right tools with the right arguments.

| Capability | Container | What it tests |
|---|---|---|
| 1 | `capability_1_bi_apis` | Tool selection and slot filling — choosing the right tool from a large set and supplying correct parameter values |
| 2 | `capability_2_dashboard_apis` | SQL query construction — 83 domains, each an independent SQLite DB exposed as REST/MCP tools |
| 3 | `capability_3_multihop_reasoning` | Multi-hop reasoning — routes to BPO analytics tools (`domain=bpo`) or SQL tools (other domains) |
| 4 | `capability_4_multiturn` | Multi-turn retrieval — combines SQL tools with a ChromaDB semantic retriever; agent must choose between lookup and vector search |

The runner connects via `docker exec` (stdio transport) — no ports need to be exposed. Domain is passed as the `MCP_DOMAIN` environment variable, which filters the tools the server exposes.

**Setup:**
```bash
make build          # build the benchmark_environ image (or: make pull)
make download       # download benchmark data into data/
docker compose up -d        # start all four capability containers
```

---

## 1. Explore tools interactively

These scripts let you poke at the environment before writing an agent.

**Connect via Docker and list tools** ([`quick_start_benchmark/simple_docker.py`](quick_start_benchmark/simple_docker.py)):
```bash
python examples/quick_start_benchmark/simple_docker.py --capability-id 2 --domain hockey
python examples/quick_start_benchmark/simple_docker.py --capability-id 3 --domain bpo
python examples/quick_start_benchmark/simple_docker.py --capability-id 1 --domain superhero
```
Opens an MCP session via `docker exec`, prints all available tools with descriptions, and includes a commented-out example showing how to call a tool directly with `session.call_tool(name, args)`.

**CLI — list tools for a capability** ([`quick_start_mcp_tools/list_tools.py`](quick_start_mcp_tools/list_tools.py)):
```bash
python examples/quick_start_mcp_tools/list_tools.py --capability-id 2 --domain hockey
python examples/quick_start_mcp_tools/list_tools.py --capability-id 2 --domain hockey --verbose   # include parameter details
python examples/quick_start_mcp_tools/list_tools.py --capability-id 2 --domain hockey --json      # machine-readable output
```

**CLI — call a specific tool** ([`quick_start_mcp_tools/invoke_tool.py`](quick_start_mcp_tools/invoke_tool.py)):
```bash
# See what tools are available first
python examples/quick_start_mcp_tools/invoke_tool.py --capability-id 2 --domain hockey --list

# Call a tool with no arguments
python examples/quick_start_mcp_tools/invoke_tool.py --capability-id 2 --domain hockey --tool <tool_name>

# Call a tool with JSON arguments
python examples/quick_start_mcp_tools/invoke_tool.py --capability-id 2 --domain hockey \
    --tool <tool_name> --args '{"param": "value"}'
```

**CLI — fetch the OpenAPI spec from a container** ([`quick_start_mcp_tools/download_spec.py`](quick_start_mcp_tools/download_spec.py)):
```bash
python examples/quick_start_mcp_tools/download_spec.py --capability-id 2           # print a summary
python examples/quick_start_mcp_tools/download_spec.py --capability-id 2 --out spec.json   # save to file
python examples/quick_start_mcp_tools/download_spec.py --capability-id 4 --port 8001       # retriever backend
```
The capability containers run a FastAPI server internally. This script fetches its `/openapi.json` via `docker exec` — useful for understanding the full API surface before the MCP layer filters it by domain.

---

## 2. Run the benchmark with your own agent

The [`quick_start_benchmark/`](quick_start_benchmark/) directory contains a benchmark runner that iterates over every domain for a capability, starts the MCP server, lists available tools, and calls a placeholder where you drop in your agent.

**Run it:**
```bash
# All domains for a capability
python examples/quick_start_benchmark/run_benchmark.py --capability 2

# Single domain smoke test
python examples/quick_start_benchmark/run_benchmark.py --capability 2 --domain hockey

# Use podman instead of docker
python examples/quick_start_benchmark/run_benchmark.py --capability 1 --runtime podman
```

**Plug in your agent** by replacing the placeholder in `run_benchmark.py`:
```python
# --- Agent call placeholder ---
# result = await agent.run(domain=domain, tools=tools, session=session)
print("\n[placeholder] Agent called with tools")
```

The `session` object is a standard `mcp.ClientSession`. Call `session.list_tools()` to discover tools and `session.call_tool(name, args)` to invoke them.

Container and command config for each capability (including per-capability domain lists) lives in [`quick_start_benchmark/server.yaml`](quick_start_benchmark/server.yaml).

---

## File map

```
examples/
├── quick_start_benchmark/
│   ├── run_benchmark.py    Benchmark runner — iterates all domains, lists tools, agent placeholder
│   ├── server.yaml         Container + command config and domain lists for all 4 capabilities
│   └── simple_docker.py    Connect via docker exec (stdio) — list and call tools interactively
└── quick_start_mcp_tools/
    ├── list_tools.py        CLI: list MCP tools for a capability + domain
    ├── invoke_tool.py       CLI: call a specific tool and print the result
    └── download_spec.py     CLI: fetch the OpenAPI spec from a capability container
```
