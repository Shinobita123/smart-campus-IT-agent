"""Uniform Cost Search over observable network facts and action effects."""

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count

from .environment import PERCEPT_KEYS, SimulatedNetwork


FAULT_GOALS = {
    "wifi": "wifi_connected",
    "dhcp_ip": "valid_ip",
    "gateway": "gateway_reachable",
    "dns": "dns_working",
    "internet": "internet_available",
}


@dataclass(frozen=True)
class Plan:
    status: str
    actions: tuple[str, ...]
    cost: int | None
    expanded_states: int
    goal_fact: str | None


def _state_key(state: dict[str, bool | None]) -> tuple[bool | None, ...]:
    return tuple(state[key] for key in PERCEPT_KEYS)


def _as_state(key: tuple[bool | None, ...]) -> dict[str, bool | None]:
    return dict(zip(PERCEPT_KEYS, key, strict=True))


def _available_for_percepts(
    state: dict[str, bool | None], excluded: frozenset[str]
) -> list[str]:
    """Model action preconditions using only known percepts."""
    actions: list[str] = []
    wifi = state["wifi_connected"]
    ip = state["valid_ip"]
    gateway = state["gateway_reachable"]

    if wifi is False:
        actions.append("reconnect_wifi")
    if wifi is True and ip is False:
        actions.append("renew_dhcp")
    if wifi is True and ip is True and gateway is False:
        actions.append("restore_gateway")
    if gateway is True and state["dns_working"] is False:
        actions.append("repair_dns")
    if gateway is True and state["internet_available"] is False:
        actions.append("restore_internet")

    local_facts = (
        "wifi_connected",
        "valid_ip",
        "gateway_reachable",
        "dns_working",
    )
    if any(state[fact] is False for fact in local_facts):
        actions.append("reset_network_stack")

    return [action for action in actions if action not in excluded]


def _apply_effect(
    state: dict[str, bool | None], action: str
) -> dict[str, bool | None]:
    successor = dict(state)
    successor.update(SimulatedNetwork.ACTION_EFFECTS[action])
    return successor


def uniform_cost_search(
    percepts: dict[str, bool | None],
    fault: str,
    available_actions: list[str],
    attempted_actions: set[str] | frozenset[str] = frozenset(),
    max_expansions: int = 1000,
) -> Plan:
    """Find the least-cost plan that repairs the diagnosed fault.

    The agent replans after every executed action. This lets it observe newly
    reachable diagnostics and react to a simulated action that did not work.
    """
    if set(percepts) != set(PERCEPT_KEYS):
        return Plan("invalid_percepts", (), None, 0, None)
    if fault not in FAULT_GOALS:
        return Plan("unsupported_goal", (), None, 0, None)

    goal_fact = FAULT_GOALS[fault]
    start = _state_key(percepts)
    start_state = _as_state(start)
    if start_state[goal_fact] is True:
        return Plan("found", (), 0, 0, goal_fact)

    initial_actions = [
        action
        for action in available_actions
        if action not in attempted_actions
    ]
    if not initial_actions:
        return Plan("no_plan", (), None, 0, goal_fact)

    sequence = count()
    frontier: list[tuple[int, int, tuple[bool | None, ...], tuple[str, ...]]] = []
    heappush(frontier, (0, next(sequence), start, ()))
    best_cost = {start: 0}
    expanded = 0

    while frontier and expanded < max_expansions:
        cost, _, state_key, path = heappop(frontier)
        if cost != best_cost.get(state_key):
            continue

        state = _as_state(state_key)
        if state[goal_fact] is True:
            return Plan("found", path, cost, expanded, goal_fact)

        expanded += 1
        excluded = frozenset(attempted_actions) | frozenset(path)
        candidates = (
            initial_actions if not path else _available_for_percepts(state, excluded)
        )

        for action in candidates:
            if action not in SimulatedNetwork.ACTION_COSTS:
                continue
            next_state = _apply_effect(state, action)
            next_key = _state_key(next_state)
            next_cost = cost + SimulatedNetwork.ACTION_COSTS[action]
            if next_cost < best_cost.get(next_key, float("inf")):
                best_cost[next_key] = next_cost
                heappush(
                    frontier,
                    (next_cost, next(sequence), next_key, path + (action,)),
                )

    status = "search_limit" if frontier else "no_plan"
    return Plan(status, (), None, expanded, goal_fact)
