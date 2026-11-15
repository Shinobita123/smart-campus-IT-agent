"""Small compatibility demo that delegates decisions to the full agent."""

import json
from pathlib import Path

from smart_campus_agent.agent import TroubleshootingAgent
from smart_campus_agent.baseline import load_case
from smart_campus_agent.environment import NetworkState


case = load_case(Path(__file__).with_name("dns_success.json"))
result = TroubleshootingAgent(case.get("max_actions", 8)).run(
    task=case["task"],
    initial_state=NetworkState(**case["initial_state"]),
    failed_actions=case["failed_actions"],
)
print(json.dumps(result, ensure_ascii=True, indent=2))
