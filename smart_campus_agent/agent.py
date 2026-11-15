"""Application-controlled loop joining diagnosis, UCS, and the simulator."""

from dataclasses import asdict

from .diagnosis import diagnose
from .environment import NetworkState, SimulatedNetwork
from .planner import uniform_cost_search


class TroubleshootingAgent:
    def __init__(self, max_actions: int = 8):
        if max_actions < 1:
            raise ValueError("max_actions must be at least 1")
        self.max_actions = max_actions

    def run(
        self,
        task: str,
        initial_state: NetworkState,
        failed_actions: tuple[str, ...] | list[str] = (),
    ) -> dict:
        """Run until verified success or a named terminal outcome.

        Ground-truth state is used only to initialize the environment and log
        simulator effects. Diagnosis and planning receive only observations.
        """
        if not isinstance(task, str) or not task.strip():
            return self._invalid_task(task)

        environment = SimulatedNetwork(initial_state, failed_actions)
        trace: list[dict] = [
            {
                "step": 0,
                "type": "input",
                "task": task,
                "initial_environment_state": asdict(initial_state),
                "note": "Initial truth configures the simulator; agent decisions use percepts only.",
            }
        ]
        selected_actions: list[dict] = []
        first_diagnosis: str | None = None
        terminal_status = "action_limit"

        for turn in range(self.max_actions + 1):
            percepts = environment.observe()
            available_actions = environment.available_actions()
            trace.append(
                {
                    "step": turn,
                    "type": "perception",
                    "percepts": percepts,
                    "available_actions": [
                        {
                            "name": action,
                            "description": SimulatedNetwork.ACTION_DESCRIPTIONS[action],
                            "cost": SimulatedNetwork.ACTION_COSTS[action],
                            "effects": SimulatedNetwork.ACTION_EFFECTS[action],
                        }
                        for action in available_actions
                    ],
                }
            )

            diagnosis = diagnose(percepts)
            diagnosis_label = (
                "multiple_faults"
                if diagnosis.status == "multiple_faults"
                else diagnosis.fault or diagnosis.status
            )
            if first_diagnosis is None:
                first_diagnosis = diagnosis_label
            trace.append(
                {
                    "step": turn,
                    "type": "diagnosis",
                    "status": diagnosis.status,
                    "fault": diagnosis.fault,
                    "explanation": diagnosis.explanation,
                    "evidence": diagnosis.evidence,
                }
            )

            if diagnosis.status == "healthy":
                terminal_status = "success"
                break
            if diagnosis.status == "multiple_faults":
                terminal_status = "unsupported_multiple_faults"
                break
            if diagnosis.status == "invalid_input":
                terminal_status = "unsupported_input"
                break
            if diagnosis.status == "inconclusive" or diagnosis.fault is None:
                terminal_status = "inconclusive"
                break
            if len(selected_actions) >= self.max_actions:
                terminal_status = "action_limit"
                break

            plan = uniform_cost_search(
                percepts=percepts,
                fault=diagnosis.fault,
                available_actions=available_actions,
                attempted_actions=environment.attempted_actions,
            )
            trace.append(
                {
                    "step": turn,
                    "type": "plan",
                    "status": plan.status,
                    "fault_goal": diagnosis.fault,
                    "goal_fact": plan.goal_fact,
                    "actions": plan.actions,
                    "estimated_cost": plan.cost,
                    "expanded_states": plan.expanded_states,
                }
            )

            if plan.status != "found" or not plan.actions:
                failed_effects = any(
                    not action["effect_applied"] for action in selected_actions
                )
                terminal_status = "recovery_failed" if failed_effects else "no_plan"
                break

            selected_action = plan.actions[0]
            if selected_action not in available_actions:
                terminal_status = "invalid_plan"
                trace.append(
                    {
                        "step": turn,
                        "type": "invalid_plan",
                        "action": selected_action,
                    }
                )
                break

            execution = environment.step(selected_action)
            execution["step"] = turn
            trace.append({"type": "execution", **execution})
            selected_actions.append(execution)
            # The loop observes again after every action, including a failed one.

        verification = environment.verify()
        success = terminal_status == "success" and verification["passed"]
        trace.append(
            {
                "step": len(selected_actions) + 1,
                "type": "verification",
                **verification,
            }
        )
        trace.append(
            {
                "step": len(selected_actions) + 2,
                "type": "terminal",
                "status": terminal_status,
                "task_success": success,
            }
        )

        total_cost = sum(action["cost"] for action in selected_actions)
        valid_count = sum(action["valid"] for action in selected_actions)
        return {
            "task": task,
            "terminal_status": terminal_status,
            "task_success": success,
            "first_diagnosis": first_diagnosis,
            "verification": verification,
            "actions_taken": [action["action"] for action in selected_actions],
            "action_count": len(selected_actions),
            "total_action_cost": total_cost,
            "valid_actions": valid_count,
            "valid_action_rate": (
                valid_count / len(selected_actions) if selected_actions else None
            ),
            "final_environment_state_for_evaluator": environment.state_for_evaluator(),
            "trace": trace,
        }

    @staticmethod
    def _invalid_task(task: object) -> dict:
        return {
            "task": task,
            "terminal_status": "unsupported_input",
            "task_success": False,
            "first_diagnosis": "invalid_input",
            "verification": {"passed": False, "percepts": {}},
            "actions_taken": [],
            "action_count": 0,
            "total_action_cost": 0,
            "valid_actions": 0,
            "valid_action_rate": None,
            "final_environment_state_for_evaluator": None,
            "trace": [
                {
                    "step": 0,
                    "type": "terminal",
                    "status": "unsupported_input",
                    "reason": "Task must be a non-empty string.",
                    "task_success": False,
                }
            ],
        }
