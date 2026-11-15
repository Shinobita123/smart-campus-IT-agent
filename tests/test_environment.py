import unittest

from smart_campus_agent.environment import NetworkState, SimulatedNetwork


class EnvironmentTests(unittest.TestCase):
    def test_dns_fault_is_observable_and_has_two_recovery_choices(self):
        environment = SimulatedNetwork(NetworkState(dns_working=False))
        self.assertFalse(environment.observe()["dns_working"])
        self.assertEqual(
            environment.available_actions(), ["repair_dns", "reset_network_stack"]
        )

    def test_downstream_percepts_are_unknown_without_wifi(self):
        environment = SimulatedNetwork(NetworkState(wifi_connected=False))
        percepts = environment.observe()
        self.assertFalse(percepts["wifi_connected"])
        self.assertIsNone(percepts["valid_ip"])
        self.assertIsNone(percepts["gateway_reachable"])
        self.assertIsNone(percepts["dns_working"])

    def test_transition_is_pure_and_step_records_effect(self):
        initial = NetworkState(dns_working=False)
        successor = SimulatedNetwork.transition(initial, "repair_dns")
        self.assertFalse(initial.dns_working)
        self.assertTrue(successor.dns_working)

        environment = SimulatedNetwork(initial)
        record = environment.step("repair_dns")
        self.assertTrue(record["valid"])
        self.assertTrue(record["effect_applied"])
        self.assertEqual(record["cost"], 1)
        self.assertTrue(environment.is_resolved())

    def test_failed_recovery_is_explicit_and_not_reoffered(self):
        environment = SimulatedNetwork(
            NetworkState(dns_working=False), failed_actions=["repair_dns"]
        )
        record = environment.step("repair_dns")
        self.assertTrue(record["valid"])
        self.assertFalse(record["effect_applied"])
        self.assertIn("reset_network_stack", environment.available_actions())
        self.assertNotIn("repair_dns", environment.available_actions())

    def test_invalid_action_is_rejected(self):
        environment = SimulatedNetwork(NetworkState(dns_working=False))
        with self.assertRaises(ValueError):
            environment.step("run_arbitrary_shell_command")


if __name__ == "__main__":
    unittest.main()
