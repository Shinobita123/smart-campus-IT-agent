"""Command-line interface for running the troubleshooting agent."""

import argparse
import json
from pathlib import Path

from .agent import TroubleshootingAgent
from .baseline import load_case, run_baseline, write_jsonl, write_summary
from .environment import NetworkState


def _load_json(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="smart-campus-agent",
        description="Run a deterministic smart-campus network troubleshooting agent.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="run one retained scenario")
    run_parser.add_argument("--case", required=True, help="scenario JSON file")

    baseline_parser = commands.add_parser(
        "baseline", help="run all cases listed in a baseline manifest"
    )
    baseline_parser.add_argument("--cases", required=True, help="baseline manifest JSON")
    baseline_parser.add_argument("--output", required=True, help="raw JSONL results path")
    baseline_parser.add_argument("--summary", required=True, help="summary JSON path")

    args = parser.parse_args()
    if args.command == "run":
        case = load_case(args.case)
        result = TroubleshootingAgent(case.get("max_actions", 8)).run(
            task=case["task"],
            initial_state=NetworkState(**case["initial_state"]),
            failed_actions=case["failed_actions"],
        )
        # Escaped JSON is safe on Windows consoles using legacy code pages.
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0

    records, summary = run_baseline(args.cases)
    write_jsonl(args.output, records)
    write_summary(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    print(f"Raw cases and traces: {args.output}")
    print(f"Summary: {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
