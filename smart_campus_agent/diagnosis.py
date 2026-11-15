from dataclasses import dataclass


@dataclass(frozen=True)
class Diagnosis:
    status: str
    fault: str | None
    explanation: str
    evidence: dict[str, bool | None]


RULES = [
    (
        "wifi",
        {"wifi_connected": False},
        "Thiết bị chưa kết nối Wi-Fi.",
    ),
    (
        "dhcp_ip",
        {"wifi_connected": True, "valid_ip": False},
        "Thiết bị chưa nhận được cấu hình IP hợp lệ.",
    ),
    (
        "gateway",
        {
            "wifi_connected": True,
            "valid_ip": True,
            "gateway_reachable": False,
        },
        "Không thể kết nối đến gateway.",
    ),
    (
        "dns",
        {
            "wifi_connected": True,
            "valid_ip": True,
            "gateway_reachable": True,
            "dns_working": False,
        },
        "DNS không phân giải được tên miền.",
    ),
    (
        "internet",
        {
            "wifi_connected": True,
            "valid_ip": True,
            "gateway_reachable": True,
            "internet_available": False,
        },
        "Không có kết nối Internet.",
    ),
]

PERCEPT_KEYS = {
    "wifi_connected",
    "valid_ip",
    "gateway_reachable",
    "dns_working",
    "internet_available",
}


def diagnose(percepts: dict[str, bool | None]) -> Diagnosis:
    if not isinstance(percepts, dict):
        return Diagnosis("invalid_input", None, "Percepts phải là dictionary.", {})

    if set(percepts) != PERCEPT_KEYS:
        return Diagnosis(
            "invalid_input",
            None,
            "Percepts thiếu hoặc có trường không được hỗ trợ.",
            dict(percepts),
        )

    if any(value is not None and not isinstance(value, bool)
           for value in percepts.values()):
        return Diagnosis(
            "invalid_input",
            None,
            "Giá trị percept phải là True, False hoặc None.",
            dict(percepts),
        )

    matches = [
        (fault, conditions, explanation)
        for fault, conditions, explanation in RULES
        if all(percepts.get(key) is value for key, value in conditions.items())
    ]

    if len(matches) > 1:
        return Diagnosis(
            "multiple_faults",
            None,
            "Có dấu hiệu của nhiều lỗi cùng lúc; cần xử lý rõ trường hợp này.",
            dict(percepts),
        )

    if len(matches) == 1:
        fault, conditions, explanation = matches[0]
        evidence = {key: percepts[key] for key in conditions}
        return Diagnosis("diagnosed", fault, explanation, evidence)

    if all(value is True for value in percepts.values()):
        return Diagnosis(
            "healthy",
            None,
            "Các kiểm tra đều thành công; không phát hiện lỗi mạng.",
            dict(percepts),
        )

    return Diagnosis(
        "inconclusive",
        None,
        "Chưa đủ thông tin để chẩn đoán lỗi được hỗ trợ.",
        dict(percepts),
    )