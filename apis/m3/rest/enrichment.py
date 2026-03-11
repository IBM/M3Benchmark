import ast
import argparse
import json
import sqlite3
import sys
from pathlib import Path

import sqlglot
from sqlglot import exp

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from agents.llm import create_llm
from langchain_core.messages import HumanMessage

DB_DIR = Path(__file__).parent.parent.parent.parent / "data" / "db"

ENRICH_PROMPT_TEMPLATE = """\
You are a QA tester. Given information about a REST API endpoint \
    and an initial natural language description of the endpoint. \
    In addition, you are given a set of sample values. These are for \
    reference only, do not include the explicit values or mention them in \
    your response. Your job is to ensure that any necessary details about \
    the format or style of the sample values is included in the description. \
    \
    For example:\
    If the sample values are ['2000/1/8', '2008/12/14', '2022/5/1', '2012/8/1', '2011/4/17']\
    then the description should include something like: 'date in YYYY/MM/DD format'

    If the sample values are ['JFK', 'PHX', 'CLE', 'SJU', 'AUS']\
    then the description should include something like: '3-letter airport ID'

If the description already includes this information, or if the format/style is obvious \
from the column name, then no change is needed. Return ONLY the enriched description, \
without any additional explanation, or the original description if no changes are needed. 

Domain: {domain}
Endpoint: {endpoint}
Operation ID: {operation_id}
Summary: {summary}
SQL: {sql}
Columns referenced: {columns}
Sample values:
{samples}

Description:"""

SERVER_DIR = Path(__file__).parent / "server"


class EndpointEnrichment:
    def __init__(self):
        self.llm = create_llm(provider="watsonx", model='mistral-large-2512')

    def discover_domains(self):
        return sorted(
            p.stem
            for p in SERVER_DIR.glob("*.py")
            if p.stem != "__init__"
        )

    def parse_query(self, sql: str) -> list[tuple[str, str]]:
        """Return (table, column) pairs for every qualified column reference in the query.

        Table aliases (e.g. T1, T2) are resolved to the actual table name.
        Unqualified column references (no table prefix) are skipped.
        Duplicate pairs are deduplicated while preserving first-seen order.
        """
        try:
            tree = sqlglot.parse_one(sql, dialect="sqlite")
        except Exception:
            return []

        # Build alias -> real table name mapping from all FROM / JOIN clauses
        alias_map: dict[str, str] = {}
        for table in tree.find_all(exp.Table):
            name = table.name
            alias = table.alias
            if alias:
                alias_map[alias] = name
            else:
                alias_map[name] = name

        # When there is only one table in the query, unqualified columns can be
        # resolved unambiguously to that table.
        sole_table = list(alias_map.values())[0] if len(alias_map) == 1 else None

        results: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for col in tree.find_all(exp.Column):
            table_ref = col.table
            col_name = col.name
            if not col_name:
                continue
            if not table_ref:
                if sole_table is None:
                    continue
                actual_table = sole_table
            else:
                actual_table = alias_map.get(table_ref, table_ref)
            pair = (actual_table, col_name)
            if pair not in seen:
                seen.add(pair)
                results.append(pair)

        return results

    def extract_queries(self, domain: str) -> list[dict]:
        source = (SERVER_DIR / f"{domain}.py").read_text()
        tree = ast.parse(source)
        results = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.AsyncFunctionDef):
                continue

            route_path = None
            operation_id = None
            summary = None
            for dec in node.decorator_list:
                if (
                    isinstance(dec, ast.Call)
                    and isinstance(dec.func, ast.Attribute)
                    and dec.func.attr == "get"
                ):
                    if dec.args:
                        route_path = ast.literal_eval(dec.args[0])
                    for kw in dec.keywords:
                        if kw.arg == "operation_id":
                            operation_id = ast.literal_eval(kw.value)
                        elif kw.arg == "summary":
                            summary = ast.literal_eval(kw.value)

            if route_path is None:
                continue

            for child in ast.walk(node):
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr == "execute"
                    and child.args
                ):
                    try:
                        sql = ast.literal_eval(child.args[0])
                    except (ValueError, TypeError):
                        sql = ast.unparse(child.args[0])
                    results.append({
                        "domain": domain,
                        "endpoint": route_path,
                        "operation_id": operation_id,
                        "summary": summary,
                        "sql": sql,
                        "columns": self.parse_query(sql),
                    })

        return results

    def enrich(self, row_dict: dict) -> dict:
        """Enrich a row by calling the LLM with endpoint metadata and sample values."""
        sample_keys = [
            k for k in row_dict
            if k not in {"domain", "endpoint", "operation_id", "summary",
                         "sql", "columns"}
        ]
        samples_str = "\n".join(
            f"  {k}: {row_dict[k][:5]}" for k in sample_keys
        )
        columns = row_dict.get("columns") or []
        prompt = ENRICH_PROMPT_TEMPLATE.format(
            domain=row_dict.get("domain", ""),
            endpoint=row_dict.get("endpoint", ""),
            operation_id=row_dict.get("operation_id", ""),
            summary=row_dict.get("summary", ""),
            sql=row_dict.get("sql", ""),
            columns=", ".join(f"{t}.{c}" for t, c in columns),
            samples=samples_str or "  (none)",
        )
        print("\n\n Prompt: ", prompt)
        response = self.llm.invoke([HumanMessage(content=prompt)])

        print("\n\nORIGINAL DESCRIPTION: ", row_dict['summary'])
        print("\nENRICHED DESCRIPTION: ", response.content)
        return {**row_dict, "llm_description": response.content}

    def view_and_add_enrichment(self, results: list[dict]) -> list[dict]:
        """For each query result, sample the first 100 values for every table/column
        pair referenced in the SQL, attach them to an enrichment dictionary, pass
        that dictionary through :func:`enrich`, and return all enriched rows.
        """
        enriched = []
        for row in results:
            domain = row["domain"]
            db_path = DB_DIR / domain / f"{domain}.sqlite"

            if not db_path.exists():
                raise FileNotFoundError(
                    f"Database not found for domain '{domain}': {db_path}"
                )

            samples: dict[str, list] = {}
            try:
                con = sqlite3.connect(str(db_path))
                cur = con.cursor()
                for table, column in row.get("columns", []):
                    key = f"{table}.{column}"
                    try:
                        cur.execute(
                            f'SELECT "{column}" FROM "{table}" LIMIT 100'
                        )
                        samples[key] = [r[0] for r in cur.fetchall()]
                    except sqlite3.Error:
                        samples[key] = []
                con.close()
            except sqlite3.Error as e:
                raise RuntimeError(
                    f"Failed to query database for domain '{domain}': {e}"
                ) from e

            row_dict = {
                "summary": row.get("summary"),
                **samples,
                **{k: v for k, v in row.items() if k != "summary"},
            }

            enriched.append(self.enrich(row_dict))

        return enriched


def main():
    parser = argparse.ArgumentParser(
        description="Print SQL query templates for REST API endpoints without substituting parameters."
    )
    parser.add_argument(
        "domains",
        nargs="*",
        default=None,
        help="Domain name(s) to inspect. Defaults to all domains.",
    )
    parser.add_argument(
        "--format",
        choices=["plain", "json", "none"],
        default="plain",
        help="Output format (default: plain)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of endpoints to process per domain (default: all)",
    )
    args = parser.parse_args()

    enrichment = EndpointEnrichment()
    domains = args.domains or enrichment.discover_domains()
    all_results = []

    for domain in domains:
        endpoints = enrichment.extract_queries(domain)
        if args.max_samples is not None:
            endpoints = endpoints[:args.max_samples]
        results = enrichment.view_and_add_enrichment(endpoints)
        all_results.extend(results)
        if args.format == "plain":
            print(f"[{domain}]")
            for r in results:
                print(f"  GET {r['endpoint']}")
                if r["summary"]:
                    print(f"    {r['summary']}")
                print(f"    {r['sql']}")
                if r["columns"]:
                    cols = ", ".join(f"{t}.{c}" for t, c in r["columns"])
                    print(f"    columns: {cols}")
                print()

    if args.format == "json":
        print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
