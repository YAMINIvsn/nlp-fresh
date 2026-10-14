
"""End-to-end DART regulatory change and impact analysis."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .obligation_extraction import extract_obligations
from .change_detection import compare_obligations
from .scope_validation import validate_change_scope
from .impact_propagation import propagate_impact
from .selective_reaudit import select_cases


def _load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON file {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Impact input JSON must contain an object.")
    return data


def _read_text(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Cannot read provision {path}: {exc}") from exc

    if not text.strip():
        raise ValueError(f"Provision text is empty: {path}")
    return text


def run_end_to_end(
    old_text: str,
    new_text: str,
    impact_input: dict,
    old_version: str = "old",
    new_version: str = "new",
    old_id: str = "provision_old",
    new_id: str = "provision_new",
    old_url: str = "",
    new_url: str = "",
    amendment_excerpt: bool = True,
) -> dict:
    """Extract, compare, map, propagate, and select cases for re-audit."""

    if not old_text.strip() or not new_text.strip():
        raise ValueError("Both old and new provision texts are required.")

    missing = {"dependencies", "cases"} - impact_input.keys()
    if missing:
        raise ValueError(f"Missing impact input keys: {sorted(missing)}")

    if not isinstance(impact_input["dependencies"], list):
        raise ValueError("'dependencies' must be a list.")
    if not isinstance(impact_input["cases"], list):
        raise ValueError("'cases' must be a list.")

    old_result = extract_obligations(old_text, old_id, old_url)
    new_result = extract_obligations(new_text, new_id, new_url)

    change_analysis = compare_obligations(
        old_result["obligations"],
        new_result["obligations"],
        old_version,
        new_version,
        amendment_excerpt=amendment_excerpt,
    )

    old_by_id = {
        item["obligation_id"]: item
        for item in old_result["obligations"]
        if item.get("obligation_id")
    }
    new_by_id = {
        item["obligation_id"]: item
        for item in new_result["obligations"]
        if item.get("obligation_id")
    }

    # Validate each proposed change's scope.
    for change in change_analysis["changes"]:
        old_item = old_by_id.get(change.get("old_obligation_id"))
        new_item = new_by_id.get(change.get("new_obligation_id"))
        original_type = change.get("change_type", "uncertain")

        scope = validate_change_scope(old_item, new_item, original_type)
        change["scope_validation"] = scope
        change["original_change_type"] = original_type

        if (
            original_type == "modified"
            and scope.get("scope_status") == "review_required"
        ):
            change["change_type"] = "uncertain"
            change.setdefault("uncertainties", []).append(
                "Scope validation requires human review."
            )

        change["uncertainties"] = list(
            dict.fromkeys(change.get("uncertainties", []))
        )

        if scope.get("needs_human_validation", True):
            change_analysis["needs_human_validation"] = True

    # Resolve extracted IDs to canonical control IDs using explicit mappings.
    mapping_config = impact_input.get("obligation_mapping", {})
    if not isinstance(mapping_config, dict):
        raise ValueError("'obligation_mapping' must be a JSON object.")

    changed_ids = set()
    unresolved_changes = []
    mapping_audit = []

    
    for change in change_analysis["changes"]:
        change_type = change.get("change_type", "uncertain")
        extracted_id = (
            change.get("new_obligation_id")
            or change.get("old_obligation_id")
        )

        # Unchanged obligations are recorded but do not initiate re-audit.
        if change_type == "unchanged":
            mapping_audit.append({
                "change_id": change.get("change_id"),
                "change_type": change_type,
                "extracted_obligation_id": extracted_id,
                "mapping_status": "not_required_for_unchanged",
                "used_for_graph_traversal": False,
                "requires_human_review": False,
            })
            continue

        # First try an exact ID match.
        mapping = mapping_config.get(str(extracted_id), {})

        # Support explicitly reviewed aliases when extraction-generated
        # IDs contain suffixes, e.g. ob1_allow_transactions.
        if not mapping and extracted_id:
            prefix = str(extracted_id).split("_", 1)[0]
            mapping = mapping_config.get(prefix, {})

        if not isinstance(mapping, dict):
            mapping = {}

        control_id = mapping.get("control_id")
        status = mapping.get("status", "unmapped")

        approved = (
            change_type in {"added", "modified", "removed"}
            and isinstance(control_id, str)
            and bool(control_id.strip())
            and status == "approved"
        )

        if approved:
            changed_ids.add(control_id.strip())
        else:
            unresolved_changes.append(change)

        mapping_audit.append({
            "change_id": change.get("change_id"),
            "change_type": change_type,
            "extracted_obligation_id": extracted_id,
            "canonical_control_id": control_id,
            "mapping_status": status if control_id else "unmapped",
            "used_for_graph_traversal": approved,
            "requires_human_review": not approved,
            "rationale": mapping.get("rationale", ""),
        })

    cases = [dict(case) for case in impact_input["cases"]]

    # A change we cannot safely map must not silently disappear.
    if unresolved_changes:
        for case in cases:
            case["uncertain"] = True

    risk_threshold = float(impact_input.get("risk_threshold", 0.35))
    if not 0.0 <= risk_threshold <= 1.0:
        raise ValueError("'risk_threshold' must be between 0 and 1.")

    if changed_ids:
        impact_result = propagate_impact(
            changed_obligation_ids=changed_ids,
            dependencies=impact_input["dependencies"],
            cases=cases,
            risk_threshold=risk_threshold,
        )
    else:
        # No approved mapping: do not invent graph relationships.
        reaudit = select_cases(
            cases=cases,
            affected_ids=set(),
            risk_threshold=risk_threshold,
        )
        impact_result = {
            "changed_obligation_ids": [],
            "affected_obligation_ids": [],
            "case_impacts": [],
            "reaudit": reaudit,
            "summary": {
                "changed_obligation_count": 0,
                "affected_obligation_count": 0,
                "potentially_affected_case_count": 0,
                "selected_for_reaudit": reaudit["counts"]["selected_cases"],
                "not_selected_for_reaudit": reaudit["counts"]["skipped_cases"],
            },
            "warning": (
                "No approved obligation-to-control mapping was available. "
                "No graph impact is asserted; unresolved changes require review."
            ),
        }

    return {
        "metadata": {
            "system": "DART",
            "pipeline": "end_to_end_regulatory_impact_analysis",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "old_version": old_version,
            "new_version": new_version,
            "comparison_mode": (
                "amendment_excerpt" if amendment_excerpt else "full_replacement"
            ),
        },
        "old_extraction": old_result,
        "new_extraction": new_result,
        "change_analysis": change_analysis,
        "obligation_mapping_audit": mapping_audit,
        "impact_analysis": impact_result,
        "unresolved_change_count": len(unresolved_changes),
        "needs_human_validation": (
            bool(change_analysis.get("needs_human_validation", False))
            or bool(unresolved_changes)
        ),
        "limitations": [
            "Mappings must be reviewed and approved before graph traversal.",
            "Dependency edges and case-control links require validation.",
            "Amendment excerpts do not prove that omitted obligations were repealed.",
            "Synthetic demo results do not establish real-world accuracy or safety.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run end-to-end DART regulatory impact analysis."
    )
    parser.add_argument("--old", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--impact-input", required=True)
    parser.add_argument("--old-version", default="old")
    parser.add_argument("--new-version", default="new")
    parser.add_argument("--old-id", default="provision_old")
    parser.add_argument("--new-id", default="provision_new")
    parser.add_argument("--old-url", default="")
    parser.add_argument("--new-url", default="")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--amendment-excerpt", action="store_true")
    mode.add_argument("--full-replacement", action="store_true")

    parser.add_argument(
        "--output",
        default="data/dart/end_to_end_output.json",
    )
    args = parser.parse_args()

    try:
        old_text = _read_text(Path(args.old))
        new_text = _read_text(Path(args.new))
        impact_input = _load_json(Path(args.impact_input))

        result = run_end_to_end(
            old_text=old_text,
            new_text=new_text,
            impact_input=impact_input,
            old_version=args.old_version,
            new_version=args.new_version,
            old_id=args.old_id,
            new_id=args.new_id,
            old_url=args.old_url,
            new_url=args.new_url,
            amendment_excerpt=not args.full_replacement,
        )

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))

    summary = result["impact_analysis"]["summary"]
    print("DART end-to-end analysis completed.")
    print(f"Report: {output_path}")
    print(f"Detected changes: {len(result['change_analysis']['changes'])}")
    print(f"Approved changed control IDs: {summary['changed_obligation_count']}")
    print(f"Potentially affected cases: {summary['potentially_affected_case_count']}")
    print(f"Selected for re-audit: {summary['selected_for_reaudit']}")
    print(f"Not selected: {summary['not_selected_for_reaudit']}")
    print(f"Unresolved changes: {result['unresolved_change_count']}")
    print(f"Human validation required: {result['needs_human_validation']}")


if __name__ == "__main__":
    main()
