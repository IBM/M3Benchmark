#!/usr/bin/env python3
"""
Simple benchmark runner for capabilities 1–4.

Iterates over every domain for the selected capability, starts the MCP server
for that domain via docker exec (stdio), lists tools, calls the agent, then
saves results in the required submission format.

Usage:
    python run_benchmark.py --capability 2                       # all domains
    python run_benchmark.py --capability 2 --domain airline      # one domain
    python run_benchmark.py --capability 2 --out results.json    # save output
    python run_benchmark.py --capability 1 --runtime podman      # use podman
    python run_benchmark.py --capability 2 --data-dir /path/to/data/test
"""

import argparse
import asyncio
import json
import shutil
import sys
import time
from pathlib import Path

import yaml
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import ValidationError

from schemas import OutputRecord, Sequence, ToolCall, Turn

PROJECT_ROOT = Path(__file__).parent
CONFIG_FILE = PROJECT_ROOT / "server.yaml"

# Default data location: <repo_root>/data/test
DEFAULT_DATA_DIR = PROJECT_ROOT.parent.parent / "data" / "test"

CAPABILITY_DIRS = {
    1: "capability_1_bi_apis",
    2: "capability_2_dashboard_apis",
    3: "capability_3_multihop_reasoning",
    4: "capability_4_multiturn",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_questions(
    data_dir: Path,
    capability: int,
    domain_filter: str | None,
) -> dict[str, list[dict]]:
    """
    Load benchmark questions from data_dir/capability_N_.../input/<domain>.json.
    Returns a dict mapping domain name -> list of question dicts with keys
    'uuid', 'domain', 'query', 'turn_id'.
    """
    cap_dir = data_dir / CAPABILITY_DIRS[capability] / "input"
    if not cap_dir.exists():
        print(f"Error: data directory not found: {cap_dir}")
        print("Run 'make download' to fetch benchmark data, or pass --data-dir.")
        sys.exit(1)

    json_files = sorted(cap_dir.glob("*.json"))
    if domain_filter:
        json_files = [f for f in json_files if f.stem == domain_filter]
        if not json_files:
            print(f"Error: no data file for domain '{domain_filter}' in {cap_dir}")
            sys.exit(1)

    questions_by_domain: dict[str, list[dict]] = {}
    for json_file in json_files:
        domain = json_file.stem
        records = json.loads(json_file.read_text(encoding="utf-8"))
        questions = []
        for rec in records:
            turns = rec.get("dialogue", {}).get("turns") or rec.get("ground_truth", [])
            if not turns:
                continue
            last = turns[-1]
            questions.append({
                "uuid": rec["uuid"],
                "domain": rec.get("domain", domain),
                "query": last["query"],
                "turn_id": last.get("turn_id", 0),
            })
        questions_by_domain[domain] = questions

    return questions_by_domain


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_runtime(preferred: str = "docker") -> str:
    if shutil.which(preferred):
        return preferred
    fallback = "podman" if preferred == "docker" else "docker"
    if shutil.which(fallback):
        print(f"  [{preferred} not found, using {fallback}]")
        return fallback
    print("Error: neither docker nor podman found on PATH.")
    sys.exit(1)


def print_tools(tools: list) -> None:
    for t in tools:
        print(f"  {t.name}")
        if t.description:
            print(f"    {t.description.strip().split(chr(10))[0]}")
        props = t.inputSchema.get("properties", {}) if t.inputSchema else {}
        required = set((t.inputSchema or {}).get("required", []))
        if props:
            print(f"    Parameters:")
            for param, schema in props.items():
                req_marker = " (required)" if param in required else ""
                ptype = schema.get("type", "any")
                desc = schema.get("description", "")
                print(f"      - {param}: {ptype}{req_marker}  {desc}")


# ---------------------------------------------------------------------------
# Per-domain runner
# ---------------------------------------------------------------------------

async def run_domain(
    domain: str,
    cfg: dict,
    rt: str,
    questions: list[dict],
) -> list[OutputRecord]:
    """
    Connect to the MCP server for `domain`, answer each question, and return
    a validated OutputRecord per question.

    Replace the placeholder agent block with your own agent implementation.
    The session object is an initialised mcp.ClientSession — call
    session.list_tools() and session.call_tool(name, args) as needed.
    """
    container = cfg["container"]
    env = cfg.get("env", {})
    command = cfg["command"]

    docker_args = ["exec", "-i"]
    for k, v in env.items():
        docker_args += ["-e", f"{k}={v}"]
    docker_args += ["-e", f"MCP_DOMAIN={domain}"]
    docker_args += [container, *command]

    params = StdioServerParameters(command=rt, args=docker_args, env=None)

    print(f"\n{'='*60}")
    print(f"Domain: {domain}  ({len(questions)} question(s))")

    results: list[OutputRecord] = []

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = (await session.list_tools()).tools
            print(f"Tools available: {len(tools)}")
            print_tools(tools)

            for q in questions:
                t0 = time.perf_counter()
                tool_calls_made: list[dict] = []

                try:
                    # ----------------------------------------------------------
                    # TODO: replace this block with your agent implementation.
                    #
                    # Your agent should:
                    #   1. Call session.call_tool(name, args) as needed.
                    #   2. Append each call to tool_calls_made:
                    #        tool_calls_made.append({"name": name, "arguments": args})
                    #   3. Return the final answer string.
                    #
                    # Example (LLM-based agent):
                    #   answer = await my_agent.run(session, q["query"])
                    # ----------------------------------------------------------
                    answer = "[TODO: agent answer]"
                    # ----------------------------------------------------------

                    record = OutputRecord(
                        uuid=q["uuid"],
                        domain=domain,
                        status="success",
                        error="",
                        duration_s=time.perf_counter() - t0,
                        output=[
                            Turn(
                                turn_id=q.get("turn_id", 0),
                                query=q["query"],
                                answer=answer,
                                sequence=Sequence(
                                    tool_call=[
                                        ToolCall(
                                            name=tc["name"],
                                            arguments=tc["arguments"],
                                        )
                                        for tc in tool_calls_made
                                    ]
                                ),
                            )
                        ],
                    )

                except ValidationError as exc:
                    # Schema violation — log and record as error so the run
                    # continues rather than crashing.
                    print(f"  [SCHEMA ERROR] uuid={q['uuid']}: {exc}")
                    record = OutputRecord(
                        uuid=q["uuid"],
                        domain=domain,
                        status="error",
                        error=f"Schema validation failed: {exc}",
                        duration_s=time.perf_counter() - t0,
                        output=[],
                    )

                except Exception as exc:
                    record = OutputRecord(
                        uuid=q["uuid"],
                        domain=domain,
                        status="error",
                        error=str(exc),
                        duration_s=time.perf_counter() - t0,
                        output=[],
                    )

                results.append(record)
                status_icon = "✓" if record.status == "success" else "✗"
                print(f"  {status_icon} {q['uuid']}  ({record.duration_s:.2f}s)")

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main(
    capability: int,
    domain_filter: str | None,
    runtime: str,
    out_path: str | None,
    data_dir: Path,
) -> None:
    config = yaml.safe_load(CONFIG_FILE.read_text())
    key = f"capability_{capability}"
    if key not in config:
        print(f"Error: capability {capability} not found in {CONFIG_FILE}.")
        sys.exit(1)

    cfg = config[key]
    rt = get_runtime(runtime)

    questions_by_domain = load_questions(data_dir, capability, domain_filter)
    domains = sorted(questions_by_domain)

    print(f"Capability {capability} | container: {cfg['container']} | domains: {len(domains)}")

    all_results: list[OutputRecord] = []
    for domain in domains:
        questions = questions_by_domain.get(domain, [])
        records = await run_domain(domain, cfg, rt, questions)
        all_results.extend(records)

    print(f"\nFinished: {len(all_results)} record(s) across {len(domains)} domain(s).")

    if out_path:
        output = json.dumps(
            [r.model_dump() for r in all_results],
            indent=2,
        )
        Path(out_path).write_text(output)
        print(f"Results saved to: {out_path}")
        print("Tip: run `python validate_output.py <file>` to verify the schema.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark MCP runner (capabilities 1–4)")
    parser.add_argument(
        "--capability", type=int, required=True, choices=[1, 2, 3, 4],
        help="Which capability to run (1, 2, 3, or 4)",
    )
    parser.add_argument("--domain", default=None, help="Run a single domain only")
    parser.add_argument("--runtime", default="docker", choices=["docker", "podman"])
    parser.add_argument(
        "--out", default=None, metavar="FILE",
        help="Save results as JSON to FILE (e.g. results/address.json)",
    )
    parser.add_argument(
        "--data-dir", default=None, metavar="DIR",
        help=f"Path to the data/test directory (default: {DEFAULT_DATA_DIR})",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else DEFAULT_DATA_DIR
    asyncio.run(main(args.capability, args.domain, args.runtime, args.out, data_dir))
