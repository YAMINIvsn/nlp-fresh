def select_hybrid(
    cases: list[dict],
    affected_ids: set[str],
    risk_threshold: float = 0.35,
) -> dict:
    """
    Hybrid DART selector.

    Outcomes:
      selected: mandatory, uncertain, or affected cases at/above threshold
      manual_triage: affected cases below threshold
      skipped: unaffected cases not requiring mandatory/uncertain review
    """
    if not 0.0 <= risk_threshold <= 1.0:
        raise ValueError("risk_threshold must be between 0 and 1")

    selected = []
    manual_triage = []
    skipped = []

    for case in cases:
        case_id = str(case.get("case_id", ""))
        obligation_value = case.get("obligation_ids", "")

        if isinstance(obligation_value, list):
            obligation_ids = {str(x).strip() for x in obligation_value if str(x).strip()}
        else:
            obligation_ids = {
                part.strip()
                for part in str(obligation_value).split(";")
                if part.strip()
            }

        affected = bool(obligation_ids & affected_ids)

        reasons = set(case.get("reasons", []))
        mandatory = (
            str(case.get("mandatory_review", "false")).lower()
            in {"1", "true", "yes"}
            or "mandatory_review" in reasons
        )
        uncertain = (
            str(case.get("uncertain", "false")).lower()
            in {"1", "true", "yes"}
            or "uncertain_case" in reasons
        )

        try:
            risk = float(case.get("risk_score", 1.0))
        except (TypeError, ValueError):
            risk = 1.0

        if not 0.0 <= risk <= 1.0:
            risk = min(1.0, max(0.0, risk))

        decision_reasons = []
        if affected:
            decision_reasons.append("depends_on_changed_obligation")
        if mandatory:
            decision_reasons.append("mandatory_review")
        if uncertain:
            decision_reasons.append("uncertain_case")
        if affected and risk >= risk_threshold:
            decision_reasons.append("risk_threshold_met")

        record = {
            "case_id": case_id,
            "risk_score": risk,
            "affected_by_graph": affected,
            "reasons": decision_reasons,
        }

        # Mandatory and uncertain cases always receive re-audit.
        if mandatory or uncertain:
            record["decision"] = "selected_for_reaudit"
            selected.append(record)
        elif affected and risk >= risk_threshold:
            record["decision"] = "selected_for_reaudit"
            selected.append(record)
        elif affected:
            record["decision"] = "manual_triage"
            record["reasons"].append("low_risk_but_graph_affected")
            manual_triage.append(record)
        else:
            record["decision"] = "skipped"
            record["reasons"].append("no_graph_impact_or_mandatory_flag")
            skipped.append(record)

    total = len(cases)
    return {
        "strategy": "Hybrid risk-aware selector",
        "risk_threshold": risk_threshold,
        "selected": selected,
        "manual_triage": manual_triage,
        "skipped": skipped,
        "counts": {
            "total_cases": total,
            "selected_cases": len(selected),
            "manual_triage_cases": len(manual_triage),
            "skipped_cases": len(skipped),
            "cases_requiring_attention": len(selected) + len(manual_triage),
            "strict_reaudit_reduction_percent": round(
                100 * len(skipped) / total, 2
            ) if total else 0.0,
            "cases_not_immediately_reaudited_percent": round(
                100 * (len(manual_triage) + len(skipped)) / total, 2
            ) if total else 0.0,
        },
        "warning": (
            "Synthetic/demo selector only. Manual triage is not a finding "
            "of compliance; evaluate against independently labelled cases."
        ),
    }
