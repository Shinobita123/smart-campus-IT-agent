import unittest

from smart_campus_agent.agent import TroubleshootingAgent
from smart_campus_agent.environment import NetworkState


class AgentTests(unittest.TestCase):
    def test_dns_vertical_slice_reaches_verified_success(self):
        result = TroubleshootingAgent().run(
            "Fix DNS so the simulated device can reach websites.",
            NetworkState(dns_working=False),
        )
        self.assertEqual(result["terminal_status"], "success")
        self.assertTrue(result["task_success"])
        self.assertEqual(result["first_diagnosis"], "dns")
        self.assertEqual(result["actions_taken"], ["repair_dns"])
        self.assertTrue(result["verification"]["passed"])

    def test_vertical_slice_resolves_each_supported_single_fault(self):
        cases = (
            (NetworkState(wifi_connected=False), "wifi"),
            (NetworkState(valid_ip=False), "dhcp_ip"),
            (NetworkState(gateway_reachable=False), "gateway"),
            (NetworkState(dns_working=False), "dns"),
            (NetworkState(internet_available=False), "internet"),
        )
        for state, expected_fault in cases:
            with self.subTest(fault=expected_fault):
                result = TroubleshootingAgent().run("Restore campus access.", state)
                self.assertEqual(result["terminal_status"], "success")
                self.assertEqual(result["first_diagnosis"], expected_fault)
                self.assertTrue(result["task_success"])

    def test_failed_recovery_terminates_after_fallbacks(self):
        result = TroubleshootingAgent().run(
            "Fix DNS.",
            NetworkState(dns_working=False),
            failed_actions=["repair_dns", "reset_network_stack"],
        )
        self.assertEqual(result["terminal_status"], "recovery_failed")
        self.assertFalse(result["task_success"])
        self.assertEqual(
            result["actions_taken"], ["repair_dns", "reset_network_stack"]
        )
        self.assertEqual(result["total_action_cost"], 6)
        self.assertFalse(result["verification"]["passed"])

    def test_multiple_faults_are_rejected_without_actions(self):
        result = TroubleshootingAgent().run(
            "Diagnose the simulated network.",
            NetworkState(dns_working=False, internet_available=False),
        )
        self.assertEqual(result["terminal_status"], "unsupported_multiple_faults")
        self.assertFalse(result["task_success"])
        self.assertEqual(result["action_count"], 0)

    def test_empty_task_is_unsupported(self):
        result = TroubleshootingAgent().run(" ", NetworkState())
        self.assertEqual(result["terminal_status"], "unsupported_input")


if __name__ == "__main__":
    unittest.main()
