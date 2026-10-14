
from unittest.mock import patch

from dart.change_detection import compare_obligations


def obligation(obligation_id, action, quote, conditions=None):
    return {
        "obligation_id": obligation_id,
        "actor": "regulated entity",
        "action": action,
        "modality": "must",
        "conditions": conditions or [],
        "exceptions": [],
        "entity_scope": "unknown",
        "effective_from": "unknown",
        "effective_to": "unknown",
        "evidence_quote": quote,
        "uncertainties": [],
    }


def make_change(
    change_type,
    old_id=None,
    new_id=None,
    old_quote="",
    new_quote="",
):
    return {
        "change_id": "C001",
        "change_type": change_type,
        "old_obligation_id": old_id,
        "new_obligation_id": new_id,
        "change_summary": "Test change",
        "old_evidence_quote": old_quote,
        "new_evidence_quote": new_quote,
        "uncertainties": [],
    }


def test_removed_becomes_uncertain_for_amendment_excerpt():
    old = [
        obligation("O1", "review documents", "Review documents.")
    ]

    fake_llm_result = {
        "changes": [
            make_change("removed", "O1", None, "Review documents.")
        ]
    }

    with patch(
        "dart.change_detection.generate_json",
        return_value=fake_llm_result,
    ):
        result = compare_obligations(
            old, [], "old", "amendment", amendment_excerpt=True
        )

    assert result["changes"][0]["change_type"] == "uncertain"
    assert result["needs_human_validation"] is True


def test_modified_with_invalid_quote_is_flagged():
    old = [
        obligation("O1", "review documents", "Review documents.")
    ]
    new = [
        obligation("N1", "review records", "Review records.")
    ]

    fake_llm_result = {
        "changes": [
            make_change(
                "modified",
                "O1",
                "N1",
                "WRONG OLD QUOTE",
                "Review records.",
            )
        ]
    }

    with patch(
        "dart.change_detection.generate_json",
        return_value=fake_llm_result,
    ):
        result = compare_obligations(
            old, new, "old", "new", amendment_excerpt=True
        )

    change = result["changes"][0]

    assert change["change_type"] == "modified"
    assert any(
        "Old evidence quote" in message
        for message in change["uncertainties"]
    )
    assert result["needs_human_validation"] is True
