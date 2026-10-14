from dart.selective_reaudit import select_cases


def test_mandatory_and_uncertain_cases_are_selected():
    cases = [
        {"case_id": "1", "obligation_ids": "A", "risk_score": "0.1",
         "mandatory_review": "true", "uncertain": "false"},
        {"case_id": "2", "obligation_ids": "B", "risk_score": "0.1",
         "mandatory_review": "false", "uncertain": "true"},
        {"case_id": "3", "obligation_ids": "C", "risk_score": "0.1",
         "mandatory_review": "false", "uncertain": "false"},
    ]
    result = select_cases(cases, {"A", "B", "C"}, risk_threshold=0.35)
    selected_ids = {x["case_id"] for x in result["selected"]}
    assert {"1", "2"} <= selected_ids
    assert "3" in {x["case_id"] for x in result["skipped"]}
