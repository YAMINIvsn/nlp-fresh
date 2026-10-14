
from dart.scope_validation import validate_change_scope


def make_obligation(scope, conditions=None, exceptions=None):
    return {
        "entity_scope": scope,
        "conditions": conditions or [],
        "exceptions": exceptions or [],
    }


def test_different_scope_flags_modified_change():
    old = make_obligation(
        "all low-risk customers",
        ["periodic KYC update"],
    )
    new = make_obligation(
        "low-risk individual customers with overdue KYC",
        ["KYC update already overdue"],
    )

    result = validate_change_scope(old, new, "modified")

    assert result["scope_status"] == "review_required"
    assert result["needs_human_validation"] is True
    assert result["issues"]


def test_same_scope_has_no_obvious_mismatch():
    old = make_obligation("all customers", ["annual review"])
    new = make_obligation("all customers", ["annual review"])

    result = validate_change_scope(old, new, "modified")

    assert result["scope_status"] == "no_obvious_scope_mismatch"
    assert result["needs_human_validation"] is True


def test_missing_new_obligation_does_not_prove_repeal():
    old = make_obligation("all customers")

    result = validate_change_scope(old, None, "removed")

    assert result["scope_status"] == "insufficient_evidence"
    assert result["needs_human_validation"] is True
