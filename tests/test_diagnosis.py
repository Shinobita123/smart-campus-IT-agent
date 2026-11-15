import unittest

from smart_campus_agent.diagnosis import diagnose


def percepts(**changes):
    values = {
        "wifi_connected": True,
        "valid_ip": True,
        "gateway_reachable": True,
        "dns_working": True,
        "internet_available": True,
    }
    values.update(changes)
    return values


class DiagnosisTests(unittest.TestCase):
    def test_each_supported_fault_category(self):
        cases = (
            (percepts(wifi_connected=False, valid_ip=None, gateway_reachable=None,
                      dns_working=None, internet_available=None), "wifi"),
            (percepts(valid_ip=False, gateway_reachable=None,
                      dns_working=None, internet_available=None), "dhcp_ip"),
            (percepts(gateway_reachable=False, dns_working=None,
                      internet_available=None), "gateway"),
            (percepts(dns_working=False), "dns"),
            (percepts(internet_available=False), "internet"),
        )
        for observed, expected in cases:
            with self.subTest(expected=expected):
                result = diagnose(observed)
                self.assertEqual(result.status, "diagnosed")
                self.assertEqual(result.fault, expected)

    def test_healthy_and_multiple_fault_states(self):
        self.assertEqual(diagnose(percepts()).status, "healthy")
        result = diagnose(percepts(dns_working=False, internet_available=False))
        self.assertEqual(result.status, "multiple_faults")

    def test_incomplete_and_invalid_percepts_are_explicit(self):
        incomplete = percepts(valid_ip=None, gateway_reachable=None,
                              dns_working=None, internet_available=None)
        self.assertEqual(diagnose(incomplete).status, "inconclusive")
        self.assertEqual(diagnose({"unexpected": True}).status, "invalid_input")
        self.assertEqual(
            diagnose(percepts(dns_working="broken")).status, "invalid_input"
        )


if __name__ == "__main__":
    unittest.main()
