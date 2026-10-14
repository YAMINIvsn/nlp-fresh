import json
from pathlib import Path

from dart.hybrid_selector import select_hybrid

SOURCE = Path("data/dart/end_to_end_simulation.json")
OUTPUT = Path("eval/results/dart_selector_comparison_v2.json")
RISK_THRESHOLD = 0.35


def main():
    report = json.loads(SOURCE.read_text(encoding="utf-8-sig"))
    impact = report["impact_analysis"]
    reaudit = impact["reaudit"]

    cases = {
        item["case_id"]: item
        for item in reaudit.get("selected", []) + reaudit.get("skipped", [])
    }

    affected = {
        item["case_id"]
        for item in impact.get("case_impacts", [])
        if item.get("impact_status") == "potentially_affected"
    }

    # Strategy 1: preserve the existing risk-based baseline.
    baseline_selected = {
        item["case_id"] for item in reaudit.get("selected", [])
    }
    all_case_ids = set(cases)

    # Strategy 2: select every graph-affected, mandatory, or uncertain case.
    impact_first_selected = set(affected)
    for case_id, case in cases.items():
        reasons = set(case.get("reasons", []))
        if "mandatory_review" in reasons or "uncertain_case" in reasons:
            impact_first_selected.add(case_id)

    # Strategy 3: hybrid selector; low-risk affected cases go to triage.
    hybrid = select_hybrid(
        list(cases.values()),
        set(impact.get("affected_obligation_ids", [])),
        risk_threshold=RISK_THRESHOLD,
    )

    # The graph's case-level impacts are the authoritative case list for
    # this comparison. Use them to derive the hybrid's affected flags too.
    # This avoids depending on obligation IDs being present in case records.
    hybrid_selected = {
        item["case_id"] for item in hybrid["selected"]
    }
    hybrid_triage = {
        item["case_id"] for item in hybrid["manual_triage"]
    }

    # Reclassify using graph-identified affected CASE IDs. This is important
    # when case rows don't contain their obligation IDs.
    hybrid_selected = set()
    hybrid_triage = set()
    hybrid_skipped = set()

    for case_id, case in cases.items():
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
        risk = min(1.0, max(0.0, risk))

        if mandatory or uncertain or (case_id in affected and risk >= RISK_THRESHOLD):
            hybrid_selected.add(case_id)
        elif case_id in affected:
            hybrid_triage.add(case_id)
        else:
            hybrid_skipped.add(case_id)

    def metrics(name, selected, triage=None):
        triage = triage or set()
        skipped = all_case_ids - selected - triage
        covered = affected & selected
        missed = affected - selected
        return {
            "strategy": name,
            "total_cases": len(all_case_ids),
            "selected_cases": len(selected),
            "manual_triage_cases": len(triage),
            "skipped_cases": len(skipped),
            "strict_reaudit_reduction_percent": round(
                100 * len(skipped) / len(all_case_ids), 2
            ) if all_case_ids else 0.0,
            "graph_affected_cases": len(affected),
            "affected_cases_selected_for_reaudit": len(covered),
            "affected_case_reaudit_coverage_percent": round(
                100 * len(covered) / len(affected), 2
            ) if affected else None,
            "affected_cases_not_selected_for_immediate_reaudit": sorted(missed),
            "affected_cases_sent_to_triage": sorted(affected & triage),
            "selected_case_ids": sorted(selected),
            "manual_triage_case_ids": sorted(triage),
            "skipped_case_ids": sorted(skipped),
        }

    results = {
        "note": (
            "Synthetic six-case demonstration. Graph-affected coverage is "
            "not recall against independently labelled regulatory violations. "
            "Manual triage is reported separately from re-audit."
        ),
        "risk_threshold": RISK_THRESHOLD,
        "strategies": {
            "baseline": metrics("Risk-based baseline", baseline_selected),
            "impact_first": metrics("Impact-first selector", impact_first_selected),
            "hybrid": metrics(
                "Hybrid risk-aware selector",
                hybrid_selected,
                hybrid_triage,
            ),
        },
        "hybrid_decisions": {
            "selected": sorted(hybrid_selected),
            "manual_triage": sorted(hybrid_triage),
            "skipped": sorted(hybrid_skipped),
        },
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("=== DART SELECTOR COMPARISON ===")
    for key, row in results["strategies"].items():
        print(f"\n{row['strategy']}")
        print(f"  Re-audit: {row['selected_cases']}/{row['total_cases']}")
        print(f"  Manual triage: {row['manual_triage_cases']}")
        print(f"  Skipped: {row['skipped_cases']}")
        print(f"  Strict re-audit reduction: {row['strict_reaudit_reduction_percent']}%")
        print(
            "  Affected cases re-audited: "
            f"{row['affected_cases_selected_for_reaudit']}/"
            f"{row['graph_affected_cases']}"
        )
        print(
            "  Affected cases not immediately re-audited: "
            f"{row['affected_cases_not_selected_for_immediate_reaudit']}"
        )

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
