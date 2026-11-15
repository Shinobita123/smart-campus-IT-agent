"""Deterministic campus-network simulator used by the agent and tests."""

from dataclasses import asdict, dataclass, replace


PERCEPT_KEYS = (
    "wifi_connected",
    "valid_ip",
    "gateway_reachable",
    "dns_working",
    "internet_available",
)


@dataclass(frozen=True)
class NetworkState:
    """Simulator state. The agent sees observations, never this object directly."""

    wifi_connected: bool = True
    valid_ip: bool = True
    gateway_reachable: bool = True
    dns_working: bool = True
    internet_available: bool = True

    def is_resolved(self) -> bool:
        return all(asdict(self).values())


class SimulatedNetwork:
    """Offline environment with inspectable actions, costs, and effects."""

    ACTION_COSTS = {
        "reconnect_wifi": 1,
        "renew_dhcp": 1,
        "restore_gateway": 2,
        "repair_dns": 1,
        "restore_internet": 4,
        "reset_network_stack": 5,
    }

    ACTION_DESCRIPTIONS = {
        "reconnect_wifi": "Reconnect the simulated device to campus Wi-Fi.",
        "renew_dhcp": "Request a valid simulated IP configuration.",
        "restore_gateway": "Restore simulated gateway reachability.",
        "repair_dns": "Restore simulated DNS resolution.",
        "restore_internet": "Restore simulated upstream Internet access.",
        "reset_network_stack": "Reset local simulated network settings.",
    }

    ACTION_EFFECTS = {
        "reconnect_wifi": {"wifi_connected": True},
        "renew_dhcp": {"valid_ip": True},
        "restore_gateway": {"gateway_reachable": True},
        "repair_dns": {"dns_working": True},
        "restore_internet": {"internet_available": True},
        # A local reset does not repair an upstream Internet outage.
        "reset_network_stack": {
            "wifi_connected": True,
            "valid_ip": True,
            "gateway_reachable": True,
            "dns_working": True,
        },
    }

    def __init__(
        self,
        initial_state: NetworkState,
        failed_actions: tuple[str, ...] | list[str] = (),
    ):
        unknown = set(failed_actions) - set(self.ACTION_COSTS)
        if unknown:
            raise ValueError(f"Unknown simulated failure actions: {sorted(unknown)}")
        self._state = initial_state
        self._failed_actions = set(failed_actions)
        self._attempted_actions: set[str] = set()

    @property
    def attempted_actions(self) -> frozenset[str]:
        return frozenset(self._attempted_actions)

    def observe(self) -> dict[str, bool | None]:
        """Return test results; downstream tests are unknown when unreachable."""
        state = self._state
        ip_result = state.valid_ip if state.wifi_connected else None
        gateway_result = state.gateway_reachable if ip_result is True else None
        can_test_services = gateway_result is True
        return {
            "wifi_connected": state.wifi_connected,
            "valid_ip": ip_result,
            "gateway_reachable": gateway_result,
            "dns_working": state.dns_working if can_test_services else None,
            "internet_available": (
                state.internet_available if can_test_services else None
            ),
        }

    @classmethod
    def actions_for(
        cls,
        state: NetworkState,
        attempted: set[str] | frozenset[str] = frozenset(),
    ) -> list[str]:
        """Pure action-availability function, also used by UCS on model states."""
        actions: list[str] = []
        if not state.wifi_connected:
            actions.append("reconnect_wifi")
        if state.wifi_connected and not state.valid_ip:
            actions.append("renew_dhcp")
        if state.wifi_connected and state.valid_ip and not state.gateway_reachable:
            actions.append("restore_gateway")
        if state.gateway_reachable and not state.dns_working:
            actions.append("repair_dns")
        if state.gateway_reachable and not state.internet_available:
            actions.append("restore_internet")

        local_problem = not all(
            (
                state.wifi_connected,
                state.valid_ip,
                state.gateway_reachable,
                state.dns_working,
            )
        )
        if local_problem:
            actions.append("reset_network_stack")
        return [action for action in actions if action not in attempted]

    def available_actions(self) -> list[str]:
        return self.actions_for(self._state, self._attempted_actions)

    @classmethod
    def transition(cls, state: NetworkState, action: str) -> NetworkState:
        """Return a hypothetical successor state without mutating the simulator."""
        if action not in cls.actions_for(state):
            raise ValueError(f"Action is not available: {action}")
        return replace(state, **cls.ACTION_EFFECTS[action])

    def step(self, action: str) -> dict:
        """Execute one valid action and return an auditable environment effect."""
        available = self.available_actions()
        if action not in available:
            raise ValueError(
                f"Action is not available: {action}; available actions: {available}"
            )

        before = self._state
        self._attempted_actions.add(action)
        failed = action in self._failed_actions
        if not failed:
            self._state = replace(self._state, **self.ACTION_EFFECTS[action])

        return {
            "action": action,
            "valid": True,
            "cost": self.ACTION_COSTS[action],
            "effect_applied": not failed,
            "failure_reason": (
                "Configured simulated recovery failure." if failed else None
            ),
            "environment_effects": self.ACTION_EFFECTS[action],
            "before": asdict(before),
            "after": asdict(self._state),
        }

    def verify(self) -> dict:
        """Run the terminal connectivity check and return its percept evidence."""
        percepts = self.observe()
        passed = all(value is True for value in percepts.values())
        return {"passed": passed, "percepts": percepts}

    def is_resolved(self) -> bool:
        return self._state.is_resolved()

    def state_for_evaluator(self) -> dict[str, bool]:
        """Expose simulator truth only to the evaluator, never to agent decisions."""
        return asdict(self._state)
