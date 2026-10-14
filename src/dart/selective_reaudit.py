"""Transparent first-pass selector. Thresholds must be tuned on validation data."""
def select_cases(cases: list[dict], affected_ids: set[str],
                 risk_threshold: float = 0.35) -> dict:
    selected, skipped = [], []
    for case in cases:
        case_id = str(case.get("case_id", ""))
        obligation_ids = {
            part.strip() for part in str(case.get("obligation_ids", "")).split(";")
            if part.strip()
        }
        affected = bool(obligation_ids & affected_ids)
        mandatory = str(case.get("mandatory_review", "false")).lower() in {"1", "true", "yes"}
        uncertain = str(case.get("uncertain", "false")).lower() in {"1", "true", "yes"}
        try:
            risk = float(case.get("risk_score", 1.0))
        except (TypeError, ValueError):
            risk = 1.0  # invalid risk -> conservative review
        risk = min(1.0, max(0.0, risk))

        reasons = []
        if affected:
            reasons.append("depends_on_changed_obligation")
        if mandatory:
            reasons.append("mandatory_review")
        if uncertain:
            reasons.append("uncertain_case")
        if affected and risk >= risk_threshold:
            reasons.append("risk_threshold_met")

        # This initial selector never skips mandatory or uncertain cases.
        should_select = mandatory or uncertain or (affected and risk >= risk_threshold)
        record = {"case_id": case_id, "risk_score": risk, "reasons": reasons}
        (selected if should_select else skipped).append(record)

    return {
        "selected": selected,
        "skipped": skipped,
        "counts": {
            "eligible_cases": len(cases),
            "selected_cases": len(selected),
            "skipped_cases": len(skipped),
            "re_audit_reduction": (
                1 - len(selected) / len(cases) if cases else 0.0
            )
        },
        "warning": (
            "This selector is a baseline, not a guarantee of safety. "
            "Evaluate missed violations against independently labelled ground truth."
        )
    }
