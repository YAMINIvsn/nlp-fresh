
"""Scope validation for regulatory obligation changes.

This module flags potentially over-broad modifications.
It does not make a final legal determination.
"""

from typing import Any


def _normalize_scope(value: Any) -> str:
    """Normalize a scope value for basic comparisons."""
    if value is None:
        return "unknown"

    return " ".join(str(value).lower().split())


def validate_change_scope(
    old_obligation: dict | None,
    new_obligation: dict | None,
    change_type: str,
) -> dict:
    """Check whether an identified change may have a scope mismatch."""

    issues = []
    warnings = []

    if change_type not in {
        "added", "removed", "modified", "unchanged", "uncertain"
    }:
        raise ValueError(f"Unsupported change type: {change_type}")

    # An added obligation has no old obligation to compare.
    if old_obligation is None:
        return {
            "scope_status": "new_obligation",
            "needs_human_validation": True,
            "issues": [],
            "warnings": [
                "Confirm whether this obligation is new or an amendment-specific exception."
            ],
        }

    # A missing new obligation does not establish repeal.
    if new_obligation is None:
        return {
            "scope_status": "insufficient_evidence",
            "needs_human_validation": True,
            "issues": [
                "No new obligation was supplied; absence alone does not establish repeal."
            ],
            "warnings": [],
        }

    old_scope = _normalize_scope(
        old_obligation.get("entity_scope", "unknown")
    )
    new_scope = _normalize_scope(
        new_obligation.get("entity_scope", "unknown")
    )

    old_conditions = {
        _normalize_scope(item)
        for item in old_obligation.get("conditions", [])
    }
    new_conditions = {
        _normalize_scope(item)
        for item in new_obligation.get("conditions", [])
    }

    old_exceptions = {
        _normalize_scope(item)
        for item in old_obligation.get("exceptions", [])
    }
    new_exceptions = {
        _normalize_scope(item)
        for item in new_obligation.get("exceptions", [])
    }

    if old_scope == "unknown" or new_scope == "unknown":
        warnings.append(
            "At least one obligation has an unknown entity scope."
        )
    elif old_scope != new_scope:
        warnings.append(
            "The old and new obligations have different stated entity scopes."
        )

    if old_conditions != new_conditions:
        warnings.append(
            "The conditions differ; verify whether the new rule applies only to a subset."
        )

    if old_exceptions != new_exceptions:
        warnings.append(
            "The exceptions differ; verify whether the obligations can coexist."
        )

    if change_type == "modified" and warnings:
        issues.append(
            "A modified classification may be too broad because scope, "
            "conditions, or exceptions differ."
        )

    if issues:
        status = "review_required"
    elif warnings:
        status = "potential_scope_difference"
    else:
        status = "no_obvious_scope_mismatch"

    return {
        "scope_status": status,
        "needs_human_validation": True,
        "issues": issues,
        "warnings": warnings,
    }
