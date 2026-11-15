"""Repeatable baseline runner and metrics for retained scenario cases."""

import json
from pathlib import Path
from typing import Any

from .agent import TroubleshootingAgent
from .environment import NetworkState


CANONICAL_COMMAND = (
    ".\\.venv\\Scripts\\python.exe -m smart_campus_agent baseline "
    "--cases data/cases/baseline.json "
    "--output results/u0_baseline.jsonl --summary results/u0_baseline_summary.json"
)


def load_case(path: str | Path) -> dict[str, Any]:
    case_path = Path(path)
    with case_path.open("r", encoding="utf-8") as stream:
        case = json.load(stream)
    required = {
        "case_id",
        "task",
        "initial_state",
        "failed_actions",
        "expected_diagnosis",
        "expected_success",
    }
    missing = required - set(case)
    if missing:
        raise ValueError(f"{case_path} is missing fields: {sorted(missing)}")
    if set(case["initial_state"]) != {
        "wifi_connected",
        "valid_ip",
        "gateway_reachable",
        "dns_working",
        "internet_available",
    }:
        raise ValueError(f"{case_path} has an invalid initial_state schema")
    if any(type(value) is not bool for value in case["initial_state"].values()):
        raise ValueError(f"{case_path} initial_state values must be booleans")
    return case


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    # The evaluator keeps labels private from the agent decision loop.
    agent = TroubleshootingAgent(max_actions=case.get("max_actions", 8))
    result = agent.run(
        task=case["task"],
        initial_state=NetworkState(**case["initial_state"]),
        failed_actions=case["failed_actions"],
    )
    diagnosis_correct = result["first_diagnosis"] == case["expected_diagnosis"]
    verification_correct = (
        result["verification"]["passed"] is case["expected_success"]
    )
    selected = result["action_count"]
    return {
        "case_id": case["case_id"],
        "case_type": case.get("case_type", "unspecified"),
        "case_input": case,
        "expected_diagnosis": case["expected_diagnosis"],
        "observed_diagnosis": result["first_diagnosis"],
        "diagnosis_correct": diagnosis_correct,
        "expected_success": case["expected_success"],
        "terminal_status": result["terminal_status"],
        "task_success_condition_met": result["task_success"],
        "verification_correct": verification_correct,
        "actions_taken": result["actions_taken"],
        "action_count": selected,
        "total_action_cost": result["total_action_cost"],
        "valid_actions": result["valid_actions"],
        "valid_action_rate": result["valid_action_rate"],
        "raw_agent_result": result,
        "selected_trace": result["trace"],
    }


def run_baseline(manifest_path: str | Path) -> tuple[list[dict], dict[str, Any]]:
    manifest_file = Path(manifest_path)
    with manifest_file.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    case_paths = manifest.get("cases")
    if not isinstance(case_paths, list) or not case_paths:
        raise ValueError("Baseline manifest must contain a non-empty 'cases' list")

    records = [evaluate_case(load_case(path)) for path in case_paths]
    count_cases = len(records)
    total_actions = sum(record["action_count"] for record in records)
    valid_actions = sum(record["valid_actions"] for record in records)
    successful = sum(record["task_success_condition_met"] for record in records)
    diagnosis_correct = sum(record["diagnosis_correct"] for record in records)
    verification_correct = sum(record["verification_correct"] for record in records)
    total_cost = sum(record["total_action_cost"] for record in records)

    summary = {
        "baseline_id": "u0-initial-measurement",
        "case_count": count_cases,
        "metrics": {
            "diagnostic_accuracy": diagnosis_correct / count_cases,
            "resolution_success_rate": successful / count_cases,
            "valid_action_rate": (
                valid_actions / total_actions if total_actions else None
            ),
            "average_action_cost_per_case": total_cost / count_cases,
            "verification_accuracy": verification_correct / count_cases,
        },
        "totals": {
            "successful_cases": successful,
            "diagnoses_correct": diagnosis_correct,
            "valid_actions": valid_actions,
            "actions_selected": total_actions,
            "total_action_cost": total_cost,
            "verification_results_correct": verification_correct,
        },
        "metric_definitions": {
            "diagnostic_accuracy": "correct first diagnosis / all cases",
            "resolution_success_rate": "cases meeting verified connectivity goal / all cases",
            "valid_action_rate": "valid executed actions / all executed actions",
            "average_action_cost_per_case": "sum of action costs / all cases, including zero-action cases",
            "verification_accuracy": "terminal verification matching evaluator ground truth / all cases",
        },
        "measurement_command": CANONICAL_COMMAND,
        "evidence_note": (
            "Each JSONL record retains the exact case input, full raw result, "
            "and execution trace."
        ),
    }
    return records, summary


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_summary(path: str | Path, summary: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
