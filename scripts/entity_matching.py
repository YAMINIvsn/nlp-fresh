"""
Entity-matching logic: checks a customer name against the sanctions
database built by build_sanctions_db.py, using fuzzy string matching
to catch spelling variations, transliteration differences, and partial
matches — since a sanctioned entity rarely appears with the exact same
spelling used in your customer's onboarding form.

This feeds the sanctions_match / unsc_list_match boolean fields that
your rule engine (KYC-008, KYC-009, FATF-009) checks.
"""

import json
from pathlib import Path
from rapidfuzz import fuzz

ENTITIES_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "sanctions_entities.json"

# Below this similarity score (0-100), we don't consider it a match.
# 85 is a reasonably strict threshold — tune this based on false
# positive/negative rates you observe during testing.
DEFAULT_MATCH_THRESHOLD = 85


def load_entities() -> list:
    with open(ENTITIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def check_sanctions_match(customer_name: str, entities: list, threshold: int = DEFAULT_MATCH_THRESHOLD) -> dict:
    """
    Check a customer name against every primary name and alias in the
    sanctions database. Returns the best match found, or None if
    nothing crosses the threshold.

    customer_name: the name to check, e.g. "Mohammed Al-Kassar"
    entities: the list loaded from sanctions_entities.json
    threshold: minimum fuzzy match score (0-100) to count as a match

    Returns:
        {
            "matched": bool,
            "best_score": float,
            "matched_entity": str or None,
            "matched_program": str or None,
        }
    """
    best_score = 0
    best_entity = None
    best_program = None

    customer_name_clean = customer_name.strip().lower()

    for entity in entities:
        # Check against the primary name
        names_to_check = [entity["primary_name"]] + entity["aliases"]
        for name in names_to_check:
            score = fuzz.token_sort_ratio(customer_name_clean, name.strip().lower())
            if score > best_score:
                best_score = score
                best_entity = name
                best_program = entity.get("program")

    return {
        "matched": best_score >= threshold,
        "best_score": best_score,
        "matched_entity": best_entity if best_score >= threshold else None,
        "matched_program": best_program if best_score >= threshold else None,
    }


if __name__ == "__main__":
    entities = load_entities()
    print(f"Loaded {len(entities)} sanctions entities for matching.\n")

    # Test cases: an exact match, a near-miss spelling, and a clean name
    test_names = [
        entities[0]["primary_name"],           # should match itself at 100
        entities[0]["primary_name"][:-2] + "e", # slight misspelling
        "John Smith Ordinary Customer",         # should not match
    ]

    for name in test_names:
        result = check_sanctions_match(name, entities)
        print(f"Checking: '{name}'")
        print(f"  -> matched={result['matched']}, score={result['best_score']:.1f}, "
              f"entity={result['matched_entity']}\n")
