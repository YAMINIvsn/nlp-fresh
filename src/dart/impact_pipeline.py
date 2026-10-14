
"""CLI for DART regulatory impact propagation."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .impact_propagation import propagate_impact


def run_impact_analysis(input_data: dict) -> dict:
    required = {"changed_obligation_ids", "dependencies", "cases"}
    missing = required - input_data.keys()

    if missing:
        raise ValueError(
            f"Missing required input fields: {sorted(missing)}"
        )

    result = propagate_impact(
        changed_obligation_ids=set(
            input_data["changed_obligation_ids"]
        ),
        dependencies=input_data["dependencies"],
        cases=input_data["cases"],
        risk_threshold=float(
            input_data.get("risk_threshold", 0.35)
        ),
    )

    result["metadata"] = {
        "system": "DART",
        "analysis_type": "regulatory_impact_propagation",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "input_regulation_version": input_data.get(
            "regulation_version", "unspecified"
        ),
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="DART regulatory impact and re-audit analysis"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8") as file:
        input_data = json.load(file)

    result = run_impact_analysis(input_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("DART impact analysis completed.")
    print(f"Report: {output_path}")
    print(
        "Potentially affected cases:",
        result["summary"]["potentially_affected_case_count"],
    )
    print(
        "Selected for re-audit:",
        result["summary"]["selected_for_reaudit"],
    )
    print(
        "Not selected:",
        result["summary"]["not_selected_for_reaudit"],
    )


if __name__ == "__main__":
    main()
