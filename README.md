# 🔷 VAKRA: A Benchmark for Evaluating Multi-Hop, Multi-Source Tool-Calling in AI Agents

**VARKA** (eValuating Agentic Knowledge Reasoning Across multi-hop, multi-source dialogues) is a tool-grounded, executable benchmark designed to evaluate how well AI agents reason end-to-end in enterprise-like settings.

Rather than testing isolated skills, **VARKA** measures compositional reasoning across APIs and documents, using full execution traces to assess whether agents can reliably complete multi-step workflows, not just individual steps. **VARKA** provides an executable environment where agents interact with over 8,000 locally hosted APIs backed by real databases spanning 62 domains, along with domain-aligned document collections.


## What VAKRA Provides

- An executable benchmark environment with 8,000+ locally hosted APIs backed by real databases across 62 domains
- Domain-aligned document collections for retrieval-augmented, cross-source reasoning
- Tasks that require 3-7 step reasoning chains across APIs, documents, and natural-language tool-use constraints
- Deterministic evaluation with live tool replay and trajectory-level verification
- Open-source code to run agents, reproduce results, and evaluate new systems end to end

## Why This Benchmark Matters

Enterprise workflows rarely look like single-turn QA or one-off function calls. In practice, agents must chain decisions across systems, reconcile mismatched schemas, interpret tool constraints expressed in natural language, ground answers in retrieved evidence, and reuse intermediate outputs to parameterize later tool calls.

VAKRA is designed to surface exactly where that reasoning succeeds or fails, including:

- entity disambiguation
- cross-source grounding
- parameter and schema alignment
- tool selection under interface variation
- policy interpretation during execution

## Benchmark Structure

VAKRA organizes evaluation into four capabilities, which together reflect three progressively harder enterprise settings.

### 1. Diverse API Interaction Styles

These tasks focus on structured tool use over APIs with different interface abstractions.

- `capability_1_bi_apis`: nested and compositional API chaining
- `capability_2_dashboard_apis`: large-scale tool selection over query-aligned endpoints

### 2. Multi-hop Reasoning over Structured APIs

These tasks require dependent reasoning chains over APIs, where earlier outputs must be interpreted and transformed for later calls.

- `capability_3_multihop_reasoning`

### 3. Multi-hop, Multi-source Reasoning with Tool-use Policies

These tasks combine APIs, document retrieval, multi-turn context, and natural-language constraints about when and how tools should be used.

- `capability_4_multiturn`

## Dataset Overview

The public dataset release is hosted on [Hugging Face](https://huggingface.co/datasets/ibm-research/VAKRA) and accompanied by a dataset card describing the task design, schema, and split statistics.

High-level test split statistics from the dataset card:

| Capability | Description | Domains | Samples |
| --- | --- | ---: | ---: |
| 1 | Nested API tool calling | 54 | 2,077 |
| 2 | Large-scale tool selection | 17 | 1,597 |
| 3 | Multi-hop reasoning | 38 | 869 |
| 4 | Multi-turn, multi-source reasoning with policies | 41 | 1,676 |

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

## Quick Start

The full setup guide lives in [setup.md](setup.md). The shortest path to a working local run is:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[init]"
pip install -r requirements_benchmark.txt

export HF_TOKEN=hf_...

make download
make pull
make start

export OPENAI_API_KEY=sk-...
python benchmark_runner.py \
  --m3_capability_id 1 \
  --domain authors \
  --max-samples-per-domain 1 \
  --provider openai
```

For local image development instead of using the prebuilt image:

```bash
make build
docker compose up -d
```

Useful follow-up docs:

- [setup.md](setup.md)
- [docs/benchmark_runner_guide.md](docs/benchmark_runner_guide.md)
- [docs/debugging.md](docs/debugging.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Running Your Own Agent

You can benchmark either the built-in runner or a custom agent implementation.

- Start from [examples/quick_start_benchmark/run_benchmark.py](examples/quick_start_benchmark/run_benchmark.py) for a minimal benchmark integration
- Use [examples/quick_start_mcp_tools/list_tools.py](examples/quick_start_mcp_tools/list_tools.py) and [examples/quick_start_mcp_tools/invoke_tool.py](examples/quick_start_mcp_tools/invoke_tool.py) to inspect and test tools directly
- See [agents/](agents/) for the built-in agent interface and components

## Evaluation

VAKRA is built for executable, verifiable evaluation. Agents interact with live local tools, and evaluation replays trajectories against those tools rather than scoring only final answers.

The evaluator combines programmatic and model-based checks to assess:

- tool-use and policy adherence
- exact matching of expected tool responses
- groundedness of the final answer with respect to tool outputs

See:

- [evaluator/README.md](evaluator/README.md)
- [evaluator/evaluator.py](evaluator/evaluator.py)

## Submitting to the Live Leaderboard

You can submit results to the public VAKRA leaderboard hosted on Hugging Face Spaces.

Submission flow:

1. Run the benchmark on the released VAKRA tasks and save your final outputs.
2. Gather the metadata for your entry, including the model name, agent setup, code or system link, and any relevant reproducibility details.
3. Open the GitHub submission template and send your results for review.

Submission links:

- Leaderboard: `https://huggingface.co/spaces/ibm-research/VAKRA`
- Submission template: `https://github.com/IBM/M3Benchmark/issues/new?template=leaderboard_submission.yml`
- Repository: `https://github.com/IBM/VAKRA`

We recommend including:

- model name and version
- agent type or prompting setup
- capability-wise scores
- code, configuration, or run details needed to reproduce the submission

## Who This Is For

VAKRA is designed for:

- researchers studying agentic reasoning, tool use, and grounding
- developers evaluating models for production-like agent workflows
- engineering teams building multi-tool enterprise assistants
- benchmark users who need reproducible, executable evaluation rather than static QA

## Public Availability

- Dataset: Hugging Face dataset release for VAKRA
- Leaderboard: `https://huggingface.co/spaces/ibm-research/VAKRA`
- Code and environment: this repository

## Acknowledgments

We especially acknowledge Hamid Adebayo, Himanshu Gupta, Renuka Sindhgatta, Sameep Mehta, Huaiyu Zhu, Chulaka Gunasekara, Jaydeep Sen, and Sara Rosenthal for their contributions and insights. We also thank our interns, Raavi Gupta and Abhinav Jain, for their efforts in benchmark generation and development.

## Citation

```
@misc{vakra-bench,
      title={VAKRA: A Benchmark for Evaluating Multi-Hop, Multi-Source Tool-Calling in AI Agents}, 
      author={Ankita Rajaram Naik*, Anupama Murthi*, Benjamin Elder*, Siyu Huo*, Praveen Venkateswaran, Danish Contractor},
      year={2026},
      url={https://huggingface.co/spaces/ibm-research/VAKRA}, 
}
```

_* Equal contributions_

## License

See [LICENSE](LICENSE) for the repository license and usage terms.
