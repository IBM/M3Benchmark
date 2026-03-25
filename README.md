# 🔷 VAKRA: A Benchmark for Evaluating Multi-Hop, Multi-Source Tool-Calling in AI Agents
**VAKRA** (e**V**aluating **A**PI and **K**nowledge **R**etrieval **A**gents using multi-hop, multi-source dialogues) is a tool-grounded, executable benchmark designed to evaluate how well AI agents reason end-to-end in enterprise-like settings.

Rather than testing isolated skills, **VAKRA** measures compositional reasoning across APIs and documents, using full execution traces to assess whether agents can reliably complete multi-step workflows, not just individual steps. **VAKRA** provides an executable environment where agents interact with over 8,000 locally hosted APIs backed by real databases spanning 62 domains, along with domain-aligned document collections.

**Resources:** [Leaderboard](https://ibm-research-vakra.hf.space) · [Dataset](https://huggingface.co/datasets/ibm-research/VAKRA) · [Blog](https://www.ibm.com/new/announcements/introducing-vakra-benchmark)

**Quick links:** [Requirements](#requirements) · [Quick Start](#quick-start) · [Exploring Available Tools](#exploring-available-tools) · [Running Your Agent](#running-your-own-agent) · [Submit to Leaderboard](#submitting-to-the-live-leaderboard)



## What VAKRA Provides

- An *executable benchmark environment* with *8,000+* locally hosted APIs backed by real databases across 62 domains
- Domain-aligned document collections for retrieval-augmented, cross-source reasoning
- Tasks that require 3-7 step reasoning chains across APIs, documents, and natural-language tool-use constraints
- Deterministic evaluation with live tool replay and trajectory-level verification
- Open-source code to run agents, reproduce results, and evaluate new systems end to end

<img width="2816" height="1536" alt="image" src="https://github.com/user-attachments/assets/d8ed707b-fbed-4157-a00d-092a9ff2a1ac" />

<div style="display: flex; align-items: center; white-space: nowrap; gap: 0.5rem; padding: 8px;">
  <div style="font-family: IBM Plex Sans; font-weight: 400; font-size: 16px; line-height: 22px; letter-spacing: 0px;">
    <a rel="noopener noreferrer" href="https://aiattribution.github.io/statements/AIA-EAI-Hin-R-?model=Gemini%203%20Flash-v1.0" data-cy="recommended-attribution-statement-text" target="_blank" style="font-family: IBM Plex Sans; font-weight: 400; font-size: 16px; line-height: 22px; letter-spacing: 0px;">AIA Entirely AI, Human-initiated, Reviewed, Gemini 3 Flash v1.0 </a>
  </div>
  <div style="display: flex; gap: 0.5rem;">
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <g clip-path="url(#clip0_50_2)">
        <path d="M12 23.5C18.3513 23.5 23.5 18.3513 23.5 12C23.5 5.64873 18.3513 0.5 12 0.5C5.64873 0.5 0.5 5.64873 0.5 12C0.5 18.3513 5.64873 23.5 12 23.5Z" fill="#4E4E4E" stroke="#161616">
        </path>
        <path d="M13.6471 15.6L13.1471 13.94H10.8171L10.3171 15.6H8.77715L11.0771 8.61998H12.9571L15.2271 15.6H13.6471ZM11.9971 9.99998H11.9471L11.1771 12.65H12.7771L11.9971 9.99998Z" fill="white">
        </path>
      </g>
      <defs>
        <clipPath id="clip0_50_2">
          <rect width="24" height="24" fill="white">
          </rect>
        </clipPath>
      </defs>
    </svg>
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M18 17H16.5V16H18V8H16.5V7H18C18.2651 7.0003 18.5193 7.10576 18.7068 7.29323C18.8942 7.4807 18.9997 7.73488 19 8V16C18.9996 16.2651 18.8942 16.5193 18.7067 16.7067C18.5193 16.8942 18.2651 16.9996 18 17Z" fill="#161616">
      </path>
      <path d="M15.5 13C16.0523 13 16.5 12.5523 16.5 12C16.5 11.4477 16.0523 11 15.5 11C14.9477 11 14.5 11.4477 14.5 12C14.5 12.5523 14.9477 13 15.5 13Z" fill="#161616">
      </path>
      <path d="M12 13C12.5523 13 13 12.5523 13 12C13 11.4477 12.5523 11 12 11C11.4477 11 11 11.4477 11 12C11 12.5523 11.4477 13 12 13Z" fill="#161616">
      </path>
      <path d="M8.5 13C9.05228 13 9.5 12.5523 9.5 12C9.5 11.4477 9.05228 11 8.5 11C7.94772 11 7.5 11.4477 7.5 12C7.5 12.5523 7.94772 13 8.5 13Z" fill="#161616">
      </path>
      <path d="M7.5 17H6C5.73488 16.9997 5.4807 16.8942 5.29323 16.7068C5.10576 16.5193 5.0003 16.2651 5 16V8C5.00026 7.73486 5.10571 7.48066 5.29319 7.29319C5.48066 7.10571 5.73486 7.00026 6 7H7.5V8H6V16H7.5V17Z" fill="#161616">
      </path>
      <circle cx="12" cy="12" r="11.5" stroke="#161616">
      </circle>
      <circle cx="12" cy="12" r="11.5" stroke="#161616">
      </circle>
    </svg>
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="12" cy="12" r="11.5" stroke="#161616">
      </circle>
      <path d="M10 6C10.4945 6 10.9778 6.14662 11.3889 6.42133C11.8 6.69603 12.1205 7.08648 12.3097 7.54329C12.4989 8.00011 12.5484 8.50277 12.452 8.98773C12.3555 9.47268 12.1174 9.91814 11.7678 10.2678C11.4181 10.6174 10.9727 10.8555 10.4877 10.952C10.0028 11.0484 9.50011 10.9989 9.04329 10.8097C8.58648 10.6205 8.19603 10.3 7.92133 9.88893C7.64662 9.4778 7.5 8.99445 7.5 8.5C7.5 7.83696 7.76339 7.20107 8.23223 6.73223C8.70107 6.26339 9.33696 6 10 6ZM10 5C9.30777 5 8.63108 5.20527 8.0555 5.58986C7.47993 5.97444 7.03133 6.52107 6.76642 7.16061C6.50151 7.80015 6.4322 8.50388 6.56725 9.18282C6.7023 9.86175 7.03564 10.4854 7.52513 10.9749C8.01461 11.4644 8.63825 11.7977 9.31718 11.9327C9.99612 12.0678 10.6999 11.9985 11.3394 11.7336C11.9789 11.4687 12.5256 11.0201 12.9101 10.4445C13.2947 9.86892 13.5 9.19223 13.5 8.5C13.5 7.57174 13.1313 6.6815 12.4749 6.02513C11.8185 5.36875 10.9283 5 10 5Z" fill="#161616">
      </path>
      <path d="M15 19H14V16.5C14 15.837 13.7366 15.2011 13.2678 14.7322C12.7989 14.2634 12.163 14 11.5 14H8.5C7.83696 14 7.20107 14.2634 6.73223 14.7322C6.26339 15.2011 6 15.837 6 16.5V19H5V16.5C5 15.5717 5.36875 14.6815 6.02513 14.0251C6.6815 13.3687 7.57174 13 8.5 13H11.5C12.4283 13 13.3185 13.3687 13.9749 14.0251C14.6313 14.6815 15 15.5717 15 16.5V19Z" fill="#161616">
      </path>
      <path d="M16.5 12.09L15.205 10.795L14.5 11.5L16.5 13.5L20 10L19.295 9.295L16.5 12.09Z" fill="#161616">
      </path>
    </svg>
  </div>
</div>

## Why This Benchmark Matters

Enterprise workflows rarely look like single-turn QA or one-off function calls. In practice, agents must chain decisions across systems, reconcile mismatched schemas, interpret tool constraints expressed in natural language, ground answers in retrieved evidence, and reuse intermediate outputs to parameterize later tool calls.

VAKRA is designed to surface exactly where that reasoning succeeds or fails, including:

- entity disambiguation
- cross-source grounding
- parameter and schema alignment
- tool selection under interface variation
- policy interpretation during execution

## Benchmark Structure

VAKRA organizes evaluation into four capabilities, which together reflect three progressively complex settings.

### 1. Diverse API Interaction Styles

These tasks focus on structured tool use over APIs with different interface abstractions.

- `capability_1_bi_apis` (API Chaining): nested and compositional API chaining
- `capability_2_dashboard_apis` (Tool Selection): large-scale tool selection over query-aligned endpoints

### 2. Multi-hop Reasoning over Structured APIs

These tasks require dependent reasoning chains over APIs, where earlier outputs must be interpreted and transformed for later calls.

We have single-turn queries that can be answered by a reasoning chain of 1–3 APIs. For example, a sample may be answered by a single API (API), or by two APIs where the output of API₁ is transformed and passed to API₂ (API₁ → API₂), or by three APIs (API₁ → API₂ → API₃).

- `capability_3_multihop_reasoning` (Multihop API Reasoning)

### 3. Multi-hop, Multi-source Reasoning with Tool-use Policies

These tasks combine reasoning over APIs and document retrieval in a multi-turn setting and also include natural-language constraints about tool use.

We have multi-turn dialogues represented as context-response-pairs wherein queries could be answered by a reasoning chain of 1-4 tools (ex., a three-turn dialogue "(API)(RAG)(API-RAG)" wherein using the context from the first two turns, an answer needs to be obtained for the (API-RAG) turn.)

- `capability_4_multiturn` (MultiHop MultiSource with Policy Adherence)

This represents the most challenging setting, mirroring decision workflows. Please find below example of all the four capabilities mentioned above.

<img width="2816" height="1536" alt="Core benchmark capabilities" src="ui/Core_benchmark_capabilities.png" />

## Dataset Overview

The public dataset release is hosted on [Hugging Face](https://huggingface.co/datasets/ibm-research/VAKRA) and accompanied by a dataset card describing the task design, schema, and split statistics.

High-level test split statistics from the dataset card:

| Capability | Description | Domains | Samples |
| --- | --- | ---: | ---: |
| 1 | API Chaining | 54 | 2,077 |
| 2 | Tool selection | 17 | 1,597 |
| 3 | Multi-hop reasoning | 38 | 869 |
| 4 | Multi-hop, multi-source reasoning with policies | 41 | 644 |

## Repository Layout

This repository includes the benchmark runtime, evaluation harness, examples, and supporting environment code.

```text
enterprise-benchmark/
├── agents/                  # Built-in agent components and wrappers
├── benchmark/               # MCP client, configs, runner helpers
├── docs/                    # Setup, architecture, runner, and debugging docs
├── environment/             # API servers, retrievers, MCP tooling
├── evaluator/               # Trajectory replay and scoring logic
├── examples/                # Quick-start examples for tools and benchmark runs
├── sample_data/             # Small example inputs/outputs
├── tests/                   # Unit, integration, and e2e tests
├── benchmark_runner.py      # Main benchmark entry point
├── benchmark_setup.py       # Setup utility / CLI entry point
├── setup.md                 # End-to-end setup guide
└── docker-compose.yml       # Container orchestration for local benchmark services
```


## Requirements

| Requirement | Details |
|---|---|
| **Python** | 3.11 or later |
| **Container runtime** | Docker (with the `docker compose` plugin) or Podman (with `podman-compose`) |
| **`make`** | Used for data download, image build, and container lifecycle targets |
| **LLM provider** | At least one of: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `WATSONX_APIKEY`, or a `LITELLM_BASE_URL` — or run [Ollama](https://ollama.com) locally (no API key required) |
| **Memory (container runtime)** | 8 GB+ allocated to Docker/Podman — capability 4 (ChromaDB) will OOM with the default 2 GB |
| **Disk space** | ~35 GB for benchmark data downloaded via `make download` |

## Quick Start

The full setup guide lives in [setup.md](setup.md). The shortest path to a working local run is:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[init]"
pip install -r requirements_benchmark.txt

make download

docker compose down # optional step
make build
docker compose up -d
```
**No API key? Try Ollama:**

```
# Install Ollama from https://ollama.com, then pull a model
ollama pull llama3.1:8b

python benchmark_runner.py \
  --capability_id 1 \
  --domain card_games \
  --max-samples-per-domain 5 \
  --provider ollama \
  --model llama3.1:8b
```


```
# Note: various providers are supported. Refer to "agents" directory for additional details.
export OPENAI_API_KEY=sk-...
python benchmark_runner.py \
  --capability_id 1 \
  --domain card_games \
  --max-samples-per-domain 5 \
  --provider openai
```



## Exploring Available Tools

Once the containers are up, you can use `tools_explorer` to browse and inspect all available tools interactively before running any benchmarks:

```bash
cd tools_explorer
uvicorn tools_explorer.app:app --reload --port 7861
```

This is useful for understanding what tools are available in a given capability and domain, inspecting their schemas, and experimenting with calls before wiring up an agent.
<img width="1302" height="727" alt="Screenshot 2026-03-24 at 6 57 31 AM" src="https://github.com/user-attachments/assets/a1fa2cce-fc68-4074-a5ba-e09e6a76c5a5" />


Useful follow-up docs:

- [setup.md](setup.md)
- [docs/benchmark_runner_guide.md](docs/benchmark_runner_guide.md)
- [docs/debugging.md](docs/debugging.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Running Your Own Agent

There are three levels of integration depending on how much you want to customise.

### Option 1 — Use `benchmark_runner.py` with a different provider or model

The built-in runner uses `LangGraphReActAgent` out of the box. Swap provider and model without touching any code:

```bash
python benchmark_runner.py \
  --capability_id 2 \
  --domain hockey \
  --provider anthropic \
  --model claude-sonnet-4-6

python benchmark_runner.py \
  --capability_id 2 \
  --domain hockey \
  --provider ollama \
  --model llama3.1:8b
```

### Option 2 — Extend the agent interface

[agents/agent_interface.py](agents/agent_interface.py) defines an `AgentInterface` ABC with a single method to implement:

```python
from agents.agent_interface import AgentInterface, AgentResponse, Message
from typing import List, Union

class MyAgent(AgentInterface):
    async def run(
        self,
        input: Union[str, List[Message]],
        additional_instructions: str = None,
    ) -> AgentResponse:
        # call your LLM / tools here
        ...
        return AgentResponse(content=answer, tool_calls=tool_calls, ...)
```

The built-in `LangGraphReActAgent` (also in [agents/agent_interface.py](agents/agent_interface.py)) is a good reference — it wraps a LangGraph ReAct loop and supports optional tool shortlisting via embedding similarity (`--top-k-tools`).

Use the `create_agent()` factory for quick instantiation without subclassing:

```python
from agents.agent_interface import create_agent

agent = create_agent(provider="ollama", model="llama3.1:8b")
```

### Option 3 — Start from the minimal example runner

[examples/quick_start_benchmark/run_benchmark.py](examples/quick_start_benchmark/run_benchmark.py) is a self-contained runner that handles MCP connection, data loading, and output formatting. It contains a clearly marked `TODO` block where you drop in your agent:

```python
# ----------------------------------------------------------
# TODO: replace this block with your agent implementation.
#
# Your agent should:
#   1. Call session.call_tool(name, args) as needed.
#   2. Append each call to tool_calls_made.
#   3. Return the final answer string.
# ----------------------------------------------------------
answer = "[TODO: agent answer]"
```

Run it with:

```bash
python examples/quick_start_benchmark/run_benchmark.py \
  --capability 2 --domain card_games --out results.json
```

To inspect available tools before writing your agent:

```bash
python examples/quick_start_mcp_tools/list_tools.py    # list tools for a capability/domain
python examples/quick_start_mcp_tools/invoke_tool.py   # call a single tool interactively
```

### Do's and don'ts

**Running tasks and domains in parallel**

| What | OK? | Notes |
|---|---|---|
| Multiple capabilities at the same time | Yes | Each capability has its own container — no conflict |
| Multiple domains for the same capability, in parallel | Careful | Runs fine, but spawns one Python process per domain. If you hit OOM inside the container, increase the container's memory limit in [docker-compose.yml](docker-compose.yml) |
| Same capability run multiple times concurrently | Careful | Same as above — extra Python processes share the container's memory budget. Increase the limit in [docker-compose.yml](docker-compose.yml) if needed |
| Multiple domains sequentially (default) | Yes | Default behaviour; domains run one after another within a single process |

To increase a container's memory limit, edit [docker-compose.yml](docker-compose.yml) and add or raise the `mem_limit` for the relevant service, then restart:

```yaml
capability_2_dashboard_apis_m3_environ:
  mem_limit: 4g   # raise as needed
```

```bash
docker compose up -d capability_2_dashboard_apis_m3_environ
```

**General do's and don'ts**

- Do run `make download` once before any benchmark run — results will silently error without the data
- Do validate output with `validate_output.py` before submitting — the evaluator will reject malformed files
- Don't share containers between different benchmark configurations — restart with `make start` if you change `docker-compose.yml`
- Don't interrupt a run mid-domain; partial domain files are valid JSON but may have fewer records than expected. Use `--resume` to continue a previous run from where it left off

## Evaluation

VAKRA is built for executable, verifiable evaluation. Agents interact with live local tools, and evaluation replays trajectories against those tools rather than scoring only final answers.

The evaluator combines programmatic and model-based checks to assess:

- tool-use and policy adherence
- exact matching of expected tool responses
- groundedness of the final answer with respect to tool outputs

See:

- [evaluator/README.md](evaluator/README.md)
- [evaluator/evaluator.py](evaluator/evaluator.py)

## Output Format and Directory Structure

### Directory layout

Output mirrors the input layout under `data/test/`. One  directory per capability:

```
data/test/                                    # input (read-only)
└── capability_2_dashboard_apis/
    └── input/
        ├── hockey.json
        ├── card_games.json
        └── ...

output/                                       # your results
└── capability_2_dashboard_apis/              # one dir per run
    ├── hockey.json                           # one file per domain
    ├── card_games.json
    ├── hockey_tools.json                     # tool-log sidecar (not for submission)
    └── run.log
```

The directory name is generated automatically as `capability_{}/`. Override it with `--output my_results/`.

### Output file schema

Each `<domain>.json` is a JSON array — one record per question:

```json
[
  {
    "uuid":       "8a751d8b-...",
    "domain":     "hockey",
    "status":     "success",
    "error":      "",
    "duration_s": 3.14,
    "output": [
      {
        "turn_id":  1,
        "query":    "How many teams played in the 2018 playoffs?",
        "answer":   "16",
        "sequence": {
          "tool_call": [
            {"name": "get_hockey_teams", "arguments": {"season": 2018}},
            {"name": "compute_data_count", "arguments": {"key_name": "team_id"}}
          ]
        }
      }
    ]
  }
]
```

If you modify the benchmark_runner.py, `*_tools.json` sidecars in the same directory record which tools were shortlisted per query — they are not part of the submission schema and are automatically skipped by `validate_output.py`.

### Validating output

```bash
# Validate all four capabilities under output/
python validate_output.py --all

# Validate a single capability (finds all output/capability_2_*/ dirs automatically)
python validate_output.py --capability 2

# Validate a specific run directory directly
python validate_output.py output/capability_2_dashboard_apis/

# Validate a single domain file
python validate_output.py output/capability_2_dashboard_apis/hockey.json

# Use a non-default output root
python validate_output.py --all --output-dir my_results/
```

`*_tools.json` sidecar files are automatically skipped — they record tool shortlisting logs and are not part of the submission schema.

## Submitting to the Live Leaderboard

You can submit results to the public VAKRA leaderboard hosted on Hugging Face Spaces.

Submission flow:

1. Run the benchmark on the released VAKRA tasks and save your final outputs.
2. Gather the metadata for your entry, including the model name, agent setup, code or system link, and any relevant reproducibility details.
3. Open the GitHub submission template and send your results for review.

Submission links:

- Leaderboard: [https://ibm-research-vakra.hf.space](https://ibm-research-vakra.hf.space)
- Submission template: [https://github.com/IBM/vakra/issues/new?template=leaderboard_submission.yml](https://github.com/IBM/vakra/issues/new?template=leaderboard_submission.yml)
- Repository: [https://github.com/IBM/vakra](https://github.com/IBM/vakra)

We recommend including:

- model name and version
- agent type or prompting setup
- capability-wise scores
- code, configuration, or run details needed to reproduce the submission

## Workflow In a Nutshell

The diagram below shows the end-to-end flow — from setup through to leaderboard submission — and marks the three points where you can plug in your own agent.
<img width="2760" height="1504" alt="Gemini_Generated_Image_wt8p2pwt8p2pwt8p" src="https://github.com/user-attachments/assets/443ead78-d133-4d4c-a74f-1c65984c1a7b" />

> **`MCP_DOMAIN`** must exactly match a domain name that exists under `data/test/capability_N_*/input/` (e.g. `hockey`, `card_games`, `airline`). The MCP server uses this value to scope its SQLite database and, for capability 4, its ChromaDB collection. Passing an unknown domain name will cause the server to fail silently or return empty results.

<div style="display: flex; align-items: center; white-space: nowrap; gap: 0.5rem; padding: 8px;">
  <div style="font-family: IBM Plex Sans; font-weight: 400; font-size: 16px; line-height: 22px; letter-spacing: 0px;">
    <a rel="noopener noreferrer" href="https://aiattribution.github.io/statements/AIA-EAI-Hin-R-?model=Gemini%203%20Flash-v1.0" data-cy="recommended-attribution-statement-text" target="_blank" style="font-family: IBM Plex Sans; font-weight: 400; font-size: 16px; line-height: 22px; letter-spacing: 0px;">AIA Entirely AI, Human-initiated, Reviewed, Gemini 3 Flash v1.0 </a>
  </div>
  <div style="display: flex; gap: 0.5rem;">
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <g clip-path="url(#clip0_50_2)">
        <path d="M12 23.5C18.3513 23.5 23.5 18.3513 23.5 12C23.5 5.64873 18.3513 0.5 12 0.5C5.64873 0.5 0.5 5.64873 0.5 12C0.5 18.3513 5.64873 23.5 12 23.5Z" fill="#4E4E4E" stroke="#161616">
        </path>
        <path d="M13.6471 15.6L13.1471 13.94H10.8171L10.3171 15.6H8.77715L11.0771 8.61998H12.9571L15.2271 15.6H13.6471ZM11.9971 9.99998H11.9471L11.1771 12.65H12.7771L11.9971 9.99998Z" fill="white">
        </path>
      </g>
      <defs>
        <clipPath id="clip0_50_2">
          <rect width="24" height="24" fill="white">
          </rect>
        </clipPath>
      </defs>
    </svg>
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M18 17H16.5V16H18V8H16.5V7H18C18.2651 7.0003 18.5193 7.10576 18.7068 7.29323C18.8942 7.4807 18.9997 7.73488 19 8V16C18.9996 16.2651 18.8942 16.5193 18.7067 16.7067C18.5193 16.8942 18.2651 16.9996 18 17Z" fill="#161616">
      </path>
      <path d="M15.5 13C16.0523 13 16.5 12.5523 16.5 12C16.5 11.4477 16.0523 11 15.5 11C14.9477 11 14.5 11.4477 14.5 12C14.5 12.5523 14.9477 13 15.5 13Z" fill="#161616">
      </path>
      <path d="M12 13C12.5523 13 13 12.5523 13 12C13 11.4477 12.5523 11 12 11C11.4477 11 11 11.4477 11 12C11 12.5523 11.4477 13 12 13Z" fill="#161616">
      </path>
      <path d="M8.5 13C9.05228 13 9.5 12.5523 9.5 12C9.5 11.4477 9.05228 11 8.5 11C7.94772 11 7.5 11.4477 7.5 12C7.5 12.5523 7.94772 13 8.5 13Z" fill="#161616">
      </path>
      <path d="M7.5 17H6C5.73488 16.9997 5.4807 16.8942 5.29323 16.7068C5.10576 16.5193 5.0003 16.2651 5 16V8C5.00026 7.73486 5.10571 7.48066 5.29319 7.29319C5.48066 7.10571 5.73486 7.00026 6 7H7.5V8H6V16H7.5V17Z" fill="#161616">
      </path>
      <circle cx="12" cy="12" r="11.5" stroke="#161616">
      </circle>
      <circle cx="12" cy="12" r="11.5" stroke="#161616">
      </circle>
    </svg>
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="12" cy="12" r="11.5" stroke="#161616">
      </circle>
      <path d="M10 6C10.4945 6 10.9778 6.14662 11.3889 6.42133C11.8 6.69603 12.1205 7.08648 12.3097 7.54329C12.4989 8.00011 12.5484 8.50277 12.452 8.98773C12.3555 9.47268 12.1174 9.91814 11.7678 10.2678C11.4181 10.6174 10.9727 10.8555 10.4877 10.952C10.0028 11.0484 9.50011 10.9989 9.04329 10.8097C8.58648 10.6205 8.19603 10.3 7.92133 9.88893C7.64662 9.4778 7.5 8.99445 7.5 8.5C7.5 7.83696 7.76339 7.20107 8.23223 6.73223C8.70107 6.26339 9.33696 6 10 6ZM10 5C9.30777 5 8.63108 5.20527 8.0555 5.58986C7.47993 5.97444 7.03133 6.52107 6.76642 7.16061C6.50151 7.80015 6.4322 8.50388 6.56725 9.18282C6.7023 9.86175 7.03564 10.4854 7.52513 10.9749C8.01461 11.4644 8.63825 11.7977 9.31718 11.9327C9.99612 12.0678 10.6999 11.9985 11.3394 11.7336C11.9789 11.4687 12.5256 11.0201 12.9101 10.4445C13.2947 9.86892 13.5 9.19223 13.5 8.5C13.5 7.57174 13.1313 6.6815 12.4749 6.02513C11.8185 5.36875 10.9283 5 10 5Z" fill="#161616">
      </path>
      <path d="M15 19H14V16.5C14 15.837 13.7366 15.2011 13.2678 14.7322C12.7989 14.2634 12.163 14 11.5 14H8.5C7.83696 14 7.20107 14.2634 6.73223 14.7322C6.26339 15.2011 6 15.837 6 16.5V19H5V16.5C5 15.5717 5.36875 14.6815 6.02513 14.0251C6.6815 13.3687 7.57174 13 8.5 13H11.5C12.4283 13 13.3185 13.3687 13.9749 14.0251C14.6313 14.6815 15 15.5717 15 16.5V19Z" fill="#161616">
      </path>
      <path d="M16.5 12.09L15.205 10.795L14.5 11.5L16.5 13.5L20 10L19.295 9.295L16.5 12.09Z" fill="#161616">
      </path>
    </svg>
  </div>
</div>

## Environment Architecture

The benchmark runner communicates with containers exclusively over MCP stdio (via `docker exec`), never over a network socket. One Docker image (`benchmark_environ`) is built and run as four named containers — one per capability. Each container hosts long-lived FastAPI background services and an on-demand MCP server process started per benchmark call.
<img width="2760" height="1504" alt="Gemini_Generated_Image_ygavx9ygavx9ygav" src="https://github.com/user-attachments/assets/f1bce426-4cbe-4940-a3a9-00027a582124" />

<div style="display: flex; align-items: center; white-space: nowrap; gap: 0.5rem; padding: 8px;">
  <div style="font-family: IBM Plex Sans; font-weight: 400; font-size: 16px; line-height: 22px; letter-spacing: 0px;">
    <a rel="noopener noreferrer" href="https://aiattribution.github.io/statements/AIA-EAI-Hin-R-?model=Gemini%203%20Flash-v1.0" data-cy="recommended-attribution-statement-text" target="_blank" style="font-family: IBM Plex Sans; font-weight: 400; font-size: 16px; line-height: 22px; letter-spacing: 0px;">AIA Entirely AI, Human-initiated, Reviewed, Gemini 3 Flash v1.0 </a>
  </div>
  <div style="display: flex; gap: 0.5rem;">
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <g clip-path="url(#clip0_50_2)">
        <path d="M12 23.5C18.3513 23.5 23.5 18.3513 23.5 12C23.5 5.64873 18.3513 0.5 12 0.5C5.64873 0.5 0.5 5.64873 0.5 12C0.5 18.3513 5.64873 23.5 12 23.5Z" fill="#4E4E4E" stroke="#161616">
        </path>
        <path d="M13.6471 15.6L13.1471 13.94H10.8171L10.3171 15.6H8.77715L11.0771 8.61998H12.9571L15.2271 15.6H13.6471ZM11.9971 9.99998H11.9471L11.1771 12.65H12.7771L11.9971 9.99998Z" fill="white">
        </path>
      </g>
      <defs>
        <clipPath id="clip0_50_2">
          <rect width="24" height="24" fill="white">
          </rect>
        </clipPath>
      </defs>
    </svg>
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M18 17H16.5V16H18V8H16.5V7H18C18.2651 7.0003 18.5193 7.10576 18.7068 7.29323C18.8942 7.4807 18.9997 7.73488 19 8V16C18.9996 16.2651 18.8942 16.5193 18.7067 16.7067C18.5193 16.8942 18.2651 16.9996 18 17Z" fill="#161616">
      </path>
      <path d="M15.5 13C16.0523 13 16.5 12.5523 16.5 12C16.5 11.4477 16.0523 11 15.5 11C14.9477 11 14.5 11.4477 14.5 12C14.5 12.5523 14.9477 13 15.5 13Z" fill="#161616">
      </path>
      <path d="M12 13C12.5523 13 13 12.5523 13 12C13 11.4477 12.5523 11 12 11C11.4477 11 11 11.4477 11 12C11 12.5523 11.4477 13 12 13Z" fill="#161616">
      </path>
      <path d="M8.5 13C9.05228 13 9.5 12.5523 9.5 12C9.5 11.4477 9.05228 11 8.5 11C7.94772 11 7.5 11.4477 7.5 12C7.5 12.5523 7.94772 13 8.5 13Z" fill="#161616">
      </path>
      <path d="M7.5 17H6C5.73488 16.9997 5.4807 16.8942 5.29323 16.7068C5.10576 16.5193 5.0003 16.2651 5 16V8C5.00026 7.73486 5.10571 7.48066 5.29319 7.29319C5.48066 7.10571 5.73486 7.00026 6 7H7.5V8H6V16H7.5V17Z" fill="#161616">
      </path>
      <circle cx="12" cy="12" r="11.5" stroke="#161616">
      </circle>
      <circle cx="12" cy="12" r="11.5" stroke="#161616">
      </circle>
    </svg>
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="12" cy="12" r="11.5" stroke="#161616">
      </circle>
      <path d="M10 6C10.4945 6 10.9778 6.14662 11.3889 6.42133C11.8 6.69603 12.1205 7.08648 12.3097 7.54329C12.4989 8.00011 12.5484 8.50277 12.452 8.98773C12.3555 9.47268 12.1174 9.91814 11.7678 10.2678C11.4181 10.6174 10.9727 10.8555 10.4877 10.952C10.0028 11.0484 9.50011 10.9989 9.04329 10.8097C8.58648 10.6205 8.19603 10.3 7.92133 9.88893C7.64662 9.4778 7.5 8.99445 7.5 8.5C7.5 7.83696 7.76339 7.20107 8.23223 6.73223C8.70107 6.26339 9.33696 6 10 6ZM10 5C9.30777 5 8.63108 5.20527 8.0555 5.58986C7.47993 5.97444 7.03133 6.52107 6.76642 7.16061C6.50151 7.80015 6.4322 8.50388 6.56725 9.18282C6.7023 9.86175 7.03564 10.4854 7.52513 10.9749C8.01461 11.4644 8.63825 11.7977 9.31718 11.9327C9.99612 12.0678 10.6999 11.9985 11.3394 11.7336C11.9789 11.4687 12.5256 11.0201 12.9101 10.4445C13.2947 9.86892 13.5 9.19223 13.5 8.5C13.5 7.57174 13.1313 6.6815 12.4749 6.02513C11.8185 5.36875 10.9283 5 10 5Z" fill="#161616">
      </path>
      <path d="M15 19H14V16.5C14 15.837 13.7366 15.2011 13.2678 14.7322C12.7989 14.2634 12.163 14 11.5 14H8.5C7.83696 14 7.20107 14.2634 6.73223 14.7322C6.26339 15.2011 6 15.837 6 16.5V19H5V16.5C5 15.5717 5.36875 14.6815 6.02513 14.0251C6.6815 13.3687 7.57174 13 8.5 13H11.5C12.4283 13 13.3185 13.3687 13.9749 14.0251C14.6313 14.6815 15 15.5717 15 16.5V19Z" fill="#161616">
      </path>
      <path d="M16.5 12.09L15.205 10.795L14.5 11.5L16.5 13.5L20 10L19.295 9.295L16.5 12.09Z" fill="#161616">
      </path>
    </svg>
  </div>
</div>


| Capability | MCP server | Data backend |
|---|---|---|
| 1 — Slot-filling / Selection | `RouterMCPServer` | SQLite (direct Python read) |
| 2 — M3 REST SQL tools | `FastAPIMCPServer` | SQLite via M3 REST :8000 |
| 3 — BPO / M3 REST router | `BPO FastMCP` or `FastAPIMCPServer` | BPO in-process / SQLite via :8000 |
| 4 — M3 REST + Retriever | `Capability4CombinedMCPServer` | SQLite via :8000 + ChromaDB via :8001 |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full per-capability stack diagrams.

## Public Availability

- Dataset: Hugging Face dataset release for VAKRA
- Leaderboard: [https://ibm-research-vakra.hf.space](https://ibm-research-vakra.hf.space)
- Code and environment: this repository

## Reporting Issues

Found a bug or have a question about the benchmark environment, runner, or evaluation? Open an issue on GitHub:

[https://github.com/IBM/vakra/issues/new](https://github.com/IBM/vakra/issues/new)

## Acknowledgments

We especially acknowledge (in alphabetical order) Chulaka Gunasekara, Hamid Adebayo, Harold Ship, Himanshu Gupta, Huaiyu Zhu, Jaydeep Sen, Nir Mashkif, Renuka Sindhgatta, Sameep Mehta, Sara Rosenthal, and Segev Shlomov for their contributions and insights. We also thank our interns, Raavi Gupta and Abhinav Jain, for their efforts in benchmark generation and development.

## Citation

```
@misc{vakra-bench,
      title={VAKRA: A Benchmark for Evaluating Multi-Hop, Multi-Source Tool-Calling in AI Agents}, 
      author={Ankita Rajaram Naik*, Anupama Murthi*, Benjamin Elder*, Siyu Huo*, Praveen Venkateswaran, Danish Contractor},
      year={2026},
      url={https://ibm-research-vakra.hf.space}, 
}
```

_* Equal contributions_

## License

This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org)
