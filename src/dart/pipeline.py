
"""DART regulatory obligation comparison pipeline."""

import argparse
import json
from pathlib import Path

from .obligation_extraction import extract_obligations
from .change_detection import compare_obligations
from .scope_validation import validate_change_scope


def main():
    parser = argparse.ArgumentParser(
        description="DART regulatory obligation comparison"
    )

    parser.add_argument("--old", required=True, help="Old provision text file")
    parser.add_argument("--new", required=True, help="New provision text file")
    parser.add_argument("--old-version", default="old")
    parser.add_argument("--new-version", default="new")
    parser.add_argument("--old-id", default="provision_old")
    parser.add_argument("--new-id", default="provision_new")
    parser.add_argument("--old-url", default="")
    parser.add_argument("--new-url", default="")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--amendment-excerpt",
        action="store_true",
        help="Treat new text as an amendment excerpt",
    )
    mode.add_argument(
        "--full-replacement",
        action="store_true",
        help="Treat new text as a complete replacement",
    )

    parser.add_argument(
        "--output",
        default="data/dart/dart_change_output.json",
        help="Output JSON path",
    )

    args = parser.parse_args()

    old_path = Path(args.old)
    new_path = Path(args.new)

    if not old_path.is_file():
        parser.error(f"Old provision file not found: {old_path}")

    if not new_path.is_file():
        parser.error(f"New provision file not found: {new_path}")

    try:
        old_text = old_path.read_text(encoding="utf-8")
        new_text = new_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        parser.error(f"Could not read provision file: {exc}")

    old_result = extract_obligations(
        old_text,
        args.old_id,
        args.old_url,
    )
    new_result = extract_obligations(
        new_text,
        args.new_id,
        args.new_url,
    )

    # Use amendment mode by default.
    is_amendment_excerpt = not args.full_replacement

    changes = compare_obligations(
        old_result["obligations"],
        new_result["obligations"],
        args.old_version,
        args.new_version,
        amendment_excerpt=is_amendment_excerpt,
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

    # Validate scope and conservatively flag potentially broad modifications.
    for change in changes["changes"]:
        old_id = change.get("old_obligation_id")
        new_id = change.get("new_obligation_id")

        old_item = old_by_id.get(old_id) if old_id else None
        new_item = new_by_id.get(new_id) if new_id else None

        original_type = change["change_type"]

        scope_result = validate_change_scope(
            old_item,
            new_item,
            original_type,
        )
        change["scope_validation"] = scope_result
        change["original_change_type"] = original_type

        if (
            original_type == "modified"
            and scope_result["scope_status"] == "review_required"
        ):
            change["change_type"] = "uncertain"
            change.setdefault("uncertainties", []).append(
                "Scope review is required because the obligations may "
                "apply to different populations or conditions."
            )

        change["uncertainties"] = list(
            dict.fromkeys(change.get("uncertainties", []))
        )

        if scope_result.get("needs_human_validation", True):
            changes["needs_human_validation"] = True

    output = {
        "old_extraction": old_result,
        "new_extraction": new_result,
        "change_analysis": changes,
    }

    output_path = Path(args.output)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as exc:
        parser.error(f"Could not save output file: {exc}")

    print(json.dumps(changes, indent=2, ensure_ascii=False))
    print(f"\nSaved complete output to: {output_path}")


if __name__ == "__main__":
    main()
