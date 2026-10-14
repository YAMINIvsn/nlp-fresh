
from dart.impact_propagation import propagate_impact


def test_transitive_impact_reaches_compliance_case():
    dependencies = [
        {"source_id": "OB1", "dependent_id": "CTRL1"},
        {"source_id": "CTRL1", "dependent_id": "CTRL2"},
    ]

    cases = [
        {
            "case_id": "CASE1",
            "obligation_ids": "CTRL2",
            "risk_score": 0.8,
        },
        {
            "case_id": "CASE2",
            "obligation_ids": "UNRELATED",
            "risk_score": 0.9,
        },
    ]

    result = propagate_impact(
        changed_obligation_ids={"OB1"},
        dependencies=dependencies,
        cases=cases,
    )

    assert set(result["affected_obligation_ids"]) == {
        "OB1", "CTRL1", "CTRL2"
    }

    assert len(result["case_impacts"]) == 1
    assert result["case_impacts"][0]["case_id"] == "CASE1"

    path = result["case_impacts"][0]["impact_paths"][0]["path"]
    assert path == ["OB1", "CTRL1", "CTRL2"]


def test_mandatory_case_is_selected_even_if_unaffected():
    result = propagate_impact(
        changed_obligation_ids={"OB1"},
        dependencies=[],
        cases=[
            {
                "case_id": "CASE1",
                "obligation_ids": "UNRELATED",
                "mandatory_review": "true",
                "risk_score": 0.1,
            }
        ],
    )

    selected_ids = {
        item["case_id"]
        for item in result["reaudit"]["selected"]
    }

    assert "CASE1" in selected_ids


def test_uncertain_case_is_selected():
    result = propagate_impact(
        changed_obligation_ids={"OB1"},
        dependencies=[],
        cases=[
            {
                "case_id": "CASE1",
                "obligation_ids": "UNRELATED",
                "uncertain": "true",
                "risk_score": 0.1,
            }
        ],
    )

    assert result["reaudit"]["counts"]["selected_cases"] == 1


def test_empty_changed_ids_are_rejected():
    try:
        propagate_impact(
            changed_obligation_ids=set(),
            dependencies=[],
            cases=[],
        )
    except ValueError:
        return

    assert False, "Expected ValueError for empty changed IDs"
