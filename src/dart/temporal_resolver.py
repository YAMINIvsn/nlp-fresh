"""Resolve which provision version is in force on a case's audit date."""
from datetime import date


def _parse_date(value):
    if value is None or str(value).strip().lower() in {"", "unknown", "none", "nan"}:
        return None
    return date.fromisoformat(str(value)[:10])


def applicable_versions(versions: list[dict], audit_date: str,
                        jurisdiction: str, entity_scope: str) -> dict:
    """Return status='resolved' only if exactly one version matches."""
    target_date = _parse_date(audit_date)
    if target_date is None:
        return {"status": "review", "reason": "Audit date is missing or invalid", "versions": []}

    matches = []
    for row in versions:
        start = _parse_date(row.get("effective_from"))
        end = _parse_date(row.get("effective_to"))
        row_jurisdiction = str(row.get("jurisdiction", "")).strip().lower()
        row_scope = str(row.get("entity_scope", "")).strip().lower()

        if start is None:
            continue  # unknown effective date cannot establish applicability
        if target_date < start or (end is not None and target_date > end):
            continue
        if row_jurisdiction not in {"all", jurisdiction.strip().lower()}:
            continue
        if row_scope not in {"all", entity_scope.strip().lower()}:
            continue
        matches.append(row)

    if len(matches) == 1:
        return {"status": "resolved", "reason": "Exactly one applicable version matched",
                "versions": matches}
    if not matches:
        return {"status": "review", "reason": "No applicable version could be established",
                "versions": []}
    return {"status": "review", "reason": "Multiple versions matched; resolve overlap",
            "versions": matches}
