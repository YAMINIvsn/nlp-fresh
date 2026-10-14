from dart.hybrid_selector import select_hybrid


def test_high_risk_affected_case_is_selected():
    result = select_hybrid(
        [{"case_id": "C1", "obligation_ids": "O1", "risk_score": 0.8}],
        {"O1"},
    )
    assert [x["case_id"] for x in result["selected"]] == ["C1"]
    assert result["manual_triage"] == []


def test_low_risk_affected_case_goes_to_manual_triage():
    result = select_hybrid(
        [{"case_id": "C1", "obligation_ids": "O1", "risk_score": 0.2}],
        {"O1"},
    )
    assert result["selected"] == []
    assert [x["case_id"] for x in result["manual_triage"]] == ["C1"]
    assert result["skipped"] == []


def test_mandatory_case_is_always_selected():
    result = select_hybrid(
        [{
            "case_id": "C1",
            "obligation_ids": "",
            "risk_score": 0.1,
            "mandatory_review": True,
        }],
        set(),
    )
    assert [x["case_id"] for x in result["selected"]] == ["C1"]


def test_uncertain_case_is_always_selected():
    result = select_hybrid(
        [{
            "case_id": "C1",
            "obligation_ids": "",
            "risk_score": 0.1,
            "uncertain": True,
        }],
        set(),
    )
    assert [x["case_id"] for x in result["selected"]] == ["C1"]


def test_unaffected_case_is_skipped():
    result = select_hybrid(
        [{"case_id": "C1", "obligation_ids": "O2", "risk_score": 0.1}],
        {"O1"},
    )
    assert result["selected"] == []
    assert result["manual_triage"] == []
    assert [x["case_id"] for x in result["skipped"]] == ["C1"]


def test_threshold_is_inclusive():
    result = select_hybrid(
        [{"case_id": "C1", "obligation_ids": "O1", "risk_score": 0.35}],
        {"O1"},
        risk_threshold=0.35,
    )
    assert [x["case_id"] for x in result["selected"]] == ["C1"]


def test_invalid_threshold_raises_error():
    import pytest

    with pytest.raises(ValueError):
        select_hybrid([], set(), risk_threshold=1.5)
