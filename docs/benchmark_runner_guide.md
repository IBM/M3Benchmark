# Benchmark Runner Guide

Step-by-step workflow for running agent benchmarks against this repo.

## Flowchart

> **View this chart:** paste it into [mermaid.live](https://mermaid.live), open this file in VSCode with the [Markdown Preview Mermaid Support](https://marketplace.visualstudio.com/items?itemName=bierner.markdown-mermaid) extension, or push to GitHub (renders natively).

```mermaid
flowchart TD
    START([🚀 Start Here]) --> PREREQ

    subgraph PREREQ["① Prerequisites"]
        P1[Docker running\n≥ 8GB memory allocated]
        P2[Python 3.8+]
        P3[API Keys ready\nHF_TOKEN + LLM provider key]
        P1 --- P2 --- P3
    end

    PREREQ --> INSTALL

    subgraph INSTALL["② Install Python Dependencies"]
        I1["python3 -m venv .venv && source .venv/bin/activate"]
        I2["pip install -e '.[init]'"]
        I3["pip install -r requirements_benchmark.txt"]
        I1 --> I2 --> I3
    end

    INSTALL --> DATA

    subgraph DATA["③ Download Benchmark Data  ~30 GB"]
        D1["export HF_TOKEN=hf_..."]
        D2["make download"]
        D1 --> D2
    end

    DATA --> DOCKER_CHOICE{Build or Pull?}

    DOCKER_CHOICE -->|Build locally| BUILD
    DOCKER_CHOICE -->|Use pre-built image| PULL

    subgraph BUILD["④a Build Image"]
        B1["make build"]
    end

    subgraph PULL["④b Pull from Docker Hub"]
        PL1["make pull"]
    end

    BUILD --> START_CONTAINERS
    PULL --> START_CONTAINERS

    subgraph START_CONTAINERS["⑤ Start 4 Capability Containers"]
        SC1["make start\nor: docker compose up -d"]
        SC2["Wait ~60 seconds for startup"]
        SC3{"docker ps shows\n4 containers?"}
        SC1 --> SC2 --> SC3
        SC3 -->|No| TROUBLESHOOT([📖 Check docs/debugging.md])
        SC3 -->|Yes| VALIDATED
    end

    VALIDATED["✅ Containers Ready"]

    VALIDATED --> EXPLORE_CHOICE{New to the benchmark?}

    EXPLORE_CHOICE -->|Yes, explore first| EXPLORE
    EXPLORE_CHOICE -->|No, go straight to running| AGENT_CHOICE

    subgraph EXPLORE["⑥ Optional: Explore Tools"]
        E1["List tools for a domain\npython examples/quick_start_mcp_tools/list_tools.py\n--capability-id 2 --domain hockey"]
        E2["Invoke a tool manually\npython examples/quick_start_mcp_tools/invoke_tool.py\n--capability-id 2 --domain hockey --tool get_teams"]
        E3["Test raw MCP connection\npython examples/quick_start_benchmark/simple_docker.py\n--capability-id 2 --domain hockey"]
        E1 --> E2 --> E3
    end

    EXPLORE --> AGENT_CHOICE

    AGENT_CHOICE{Using built-in\nor custom agent?}

    AGENT_CHOICE -->|Built-in LangGraph ReAct| BUILTIN
    AGENT_CHOICE -->|Custom agent| CUSTOM

    subgraph BUILTIN["⑦a Use Built-in Agent"]
        BI1["Smoke test — 1 sample\npython benchmark_runner.py\n--m3_capability_id 1\n--domain california_schools\n--max-samples-per-domain 1\n--provider openai"]
        BI2["Full run\npython benchmark_runner.py\n--m3_capability_id 1 2 3 4\n--provider openai --model gpt-4o"]
        BI1 --> BI2
    end

    subgraph CUSTOM["⑦b Plug In Custom Agent"]
        CA1["Copy example template\nexamples/quick_start_benchmark/run_benchmark.py"]
        CA2["Implement the agent block\nasync with session: ...\n  tools = await session.list_tools()\n  answer = your_agent(query, tools, session)"]
        CA3["Run your custom runner\npython your_runner.py --capability 2"]
        CA1 --> CA2 --> CA3
    end

    BUILTIN --> RESULTS
    CUSTOM --> RESULTS

    subgraph RESULTS["⑧ Analyze Results"]
        R1["Output written to\noutput/capability_N_timestamp/domain.json"]
        R2["Each record contains:\n• answer  • tool_calls\n• trajectory  • status / error  • duration_s"]
        R1 --> R2
    end

    RESULTS --> DONE([🏁 Done!])

    style START fill:#4CAF50,color:#fff
    style DONE fill:#4CAF50,color:#fff
    style VALIDATED fill:#2196F3,color:#fff
    style TROUBLESHOOT fill:#FF5722,color:#fff
    style PREREQ fill:#FFF9C4
    style INSTALL fill:#FFF9C4
    style DATA fill:#FFF9C4
    style BUILD fill:#E8F5E9
    style PULL fill:#E8F5E9
    style START_CONTAINERS fill:#E3F2FD
    style EXPLORE fill:#F3E5F5
    style BUILTIN fill:#E0F7FA
    style CUSTOM fill:#E0F7FA
    style RESULTS fill:#FBE9E7
```

---

## Steps at a Glance

| Step | Action | Command |
|------|--------|---------|
| ① | Prerequisites | Docker ≥ 8 GB, Python 3.8+, API keys |
| ② | Install deps | `pip install -e '.[init]' && pip install -r requirements_benchmark.txt` |
| ③ | Download data | `export HF_TOKEN=hf_... && make download` |
| ④ | Get Docker image | `make build` or `make pull` |
| ⑤ | Start containers | `make start` → wait 60s → `docker ps` |
| ⑥ | (Optional) Explore | `list_tools.py`, `invoke_tool.py`, `simple_docker.py` |
| ⑦ | Run benchmark | `benchmark_runner.py` or your custom runner |
| ⑧ | Analyze results | `output/capability_N_timestamp/domain.json` |

---

## Capability Reference

| Capability | What it tests | Key domains |
|---|---|---|
| **1** | Tool selection & slot filling | `california_schools`, `hockey` |
| **2** | SQL query construction via REST | `hockey`, `address` |
| **3** | Multi-hop (BPO + SQL routing) | `address` |
| **4** | Multi-turn with semantic search | any |

---

## Common `benchmark_runner.py` Flags

```bash
# Smoke test — 1 sample, 1 domain
python benchmark_runner.py --m3_capability_id 1 --domain california_schools --max-samples-per-domain 1 --provider openai

# Full run across all capabilities
python benchmark_runner.py --m3_capability_id 1 2 3 4 --provider openai --model gpt-4o

# Run capabilities in parallel
python benchmark_runner.py --m3_capability_id 2 4 --parallel

# Use Anthropic instead
python benchmark_runner.py --m3_capability_id 2 --provider anthropic --model claude-sonnet-4-5-20250929

# Enable top-k tool shortlisting
python benchmark_runner.py --m3_capability_id 2 --top-k-tools 10

# Custom output directory
python benchmark_runner.py --m3_capability_id 2 --output my_results/
```

---

## Custom Agent Integration

Copy [`examples/quick_start_benchmark/run_benchmark.py`](../examples/quick_start_benchmark/run_benchmark.py) and replace the agent placeholder:

```python
async with stdio_client(params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        tools = (await session.list_tools()).tools

        # Replace this block with your agent
        answer = await your_agent(query=item.query, tools=tools, session=session)
```

Your agent receives:
- `item.query` — the natural language question
- `tools` — list of MCP tool definitions
- `session` — live MCP session; call `session.call_tool(name, args)` to invoke tools

---

## Where to View the Flowchart

| Option | How |
|--------|-----|
| **Online (easiest)** | Paste the Mermaid block into [mermaid.live](https://mermaid.live) |
| **VSCode** | Install [Markdown Preview Mermaid Support](https://marketplace.visualstudio.com/items?itemName=bierner.markdown-mermaid), then open preview (`Cmd+Shift+V`) |
| **GitHub** | Push this file — GitHub renders Mermaid natively in `.md` files |
| **JetBrains IDEs** | Install the [Mermaid plugin](https://plugins.jetbrains.com/plugin/20146-mermaid) |
