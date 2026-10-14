
from dart.change_detection import compare_obligations


def make_obligation(obligation_id, action, quote):
    return {
        "obligation_id": obligation_id,
        "actor": "regulated entity",
        "action": action,
        "modality": "must",
        "conditions": [],
        "exceptions": [],
        "entity_scope": "unknown",
        "effective_from": "unknown",
        "effective_to": "unknown",
        "evidence_quote": quote,
        "uncertainties": [],
    }


def test_amendment_excerpt_does_not_remove_old_obligations():
    old_obligations = [
        make_obligation(
            "O1",
            "review customer identification documents",
            "The entity must review customer identification documents."
        ),
        make_obligation(
            "O2",
            "screen customers against sanctions lists",
            "The entity must screen customers against sanctions lists."
        ),
        make_obligation(
            "O3",
            "verify customer identity",
            "The entity must verify customer identity."
        ),
    ]

    # This is only a newly inserted clause, not the full regulation.
    new_obligations = [
        make_obligation(
            "N1",
            "allow transactions for eligible low-risk customers",
            "The entity must allow transactions for eligible low-risk customers."
        )
    ]

    result = compare_obligations(
        old_obligations,
        new_obligations,
        "old-baseline",
        "new-amendment-excerpt",
    )

    changes = result["changes"]

    # The existing rules must not be declared removed just because
    # they are absent from the amendment excerpt.
    removed = [
        c for c in changes if c["change_type"] == "removed"
    ]

    assert removed == [], (
        "False removals detected: "
        + str(removed)
    )

    assert result["needs_human_validation"] is True
