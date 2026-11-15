import unittest

from smart_campus_agent.environment import NetworkState, SimulatedNetwork
from smart_campus_agent.planner import uniform_cost_search


class PlannerTests(unittest.TestCase):
    def test_ucs_selects_lowest_cost_dns_repair(self):
        environment = SimulatedNetwork(NetworkState(dns_working=False))
        plan = uniform_cost_search(
            environment.observe(),
            "dns",
            environment.available_actions(),
        )
        self.assertEqual(plan.status, "found")
        self.assertEqual(plan.actions, ("repair_dns",))
        self.assertEqual(plan.cost, 1)
        self.assertGreater(plan.expanded_states, 0)

    def test_ucs_chooses_direct_wifi_repair_with_partial_percepts(self):
        environment = SimulatedNetwork(NetworkState(wifi_connected=False))
        plan = uniform_cost_search(
            environment.observe(),
            "wifi",
            environment.available_actions(),
        )
        self.assertEqual(plan.actions, ("reconnect_wifi",))
        self.assertEqual(plan.cost, 1)

    def test_ucs_plans_for_each_supported_fault_category(self):
        scenarios = (
            (NetworkState(wifi_connected=False), "wifi", "reconnect_wifi", 1),
            (NetworkState(valid_ip=False), "dhcp_ip", "renew_dhcp", 1),
            (NetworkState(gateway_reachable=False), "gateway", "restore_gateway", 2),
            (NetworkState(dns_working=False), "dns", "repair_dns", 1),
            (NetworkState(internet_available=False), "internet", "restore_internet", 4),
        )
        for state, fault, expected_action, expected_cost in scenarios:
            with self.subTest(fault=fault):
                environment = SimulatedNetwork(state)
                plan = uniform_cost_search(
                    environment.observe(), fault, environment.available_actions()
                )
                self.assertEqual(plan.status, "found")
                self.assertEqual(plan.actions, (expected_action,))
                self.assertEqual(plan.cost, expected_cost)

    def test_ucs_reports_no_plan_after_all_dns_options_attempted(self):
        environment = SimulatedNetwork(NetworkState(dns_working=False))
        plan = uniform_cost_search(
            environment.observe(),
            "dns",
            environment.available_actions(),
            attempted_actions={"repair_dns", "reset_network_stack"},
        )
        self.assertEqual(plan.status, "no_plan")

    def test_unsupported_goal_is_reported(self):
        environment = SimulatedNetwork(NetworkState(dns_working=False))
        plan = uniform_cost_search(
            environment.observe(),
            "printer",
            environment.available_actions(),
        )
        self.assertEqual(plan.status, "unsupported_goal")


if __name__ == "__main__":
    unittest.main()
