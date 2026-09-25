"""
Baseline 1: Rule-Engine-Only Evaluation

The rule engine operates on STRUCTURED fields (booleans, numbers), not
raw clause text — that's a fundamentally different input type from the
text-based kyc_clauses.json dataset used for the LLM baseline. This is
expected: in the full pipeline, structured fields would come from an
NER/field-extraction step run over a real document. For this baseline,
we build a small set of structured test cases with known ground truth
(Compliant / Non-Compliant), matching the granularity of a single
rule-check rather than a whole document.

This script measures how well the rule engine alone does at flagging
violations, establishing your first row in the results comparison table.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rule_engine import load_rules, evaluate_rules

# Structured test cases with ground truth labels. Each case represents
# one "checkable unit" (analogous to what a real field-extraction layer
# would produce from a document). ground_truth is what a human compliance
# reviewer would conclude given these exact facts.
TEST_CASES = [
    {"case": {"customer_name_verified": True, "cdd_completed": True}, "ground_truth": "Compliant"},
    {"case": {"customer_name_verified": False}, "ground_truth": "Non-Compliant"},
    {"case": {"transaction_amount": 60000, "is_walkin": True, "identity_verified": False}, "ground_truth": "Non-Compliant"},
    {"case": {"transaction_amount": 20000, "is_walkin": True, "identity_verified": False}, "ground_truth": "Compliant"},
    {"case": {"aadhaar_present": True, "pan_present": False, "form60_present": False}, "ground_truth": "Compliant"},
    {"case": {"aadhaar_present": False, "pan_present": False, "form60_present": False}, "ground_truth": "Non-Compliant"},
    {"case": {"risk_category": "high"}, "ground_truth": "Compliant"},
    {"case": {"risk_category": None}, "ground_truth": "Non-Compliant"},
    {"case": {"is_pep": True, "senior_approval": True, "source_of_funds_verified": True}, "ground_truth": "Compliant"},
    {"case": {"is_pep": True, "senior_approval": False, "source_of_funds_verified": False}, "ground_truth": "Non-Compliant"},
    {"case": {"suspicious_flag": True, "str_filed": True}, "ground_truth": "Compliant"},
    {"case": {"suspicious_flag": True, "str_filed": False}, "ground_truth": "Non-Compliant"},
    {"case": {"sanctions_match": False}, "ground_truth": "Compliant"},
    {"case": {"sanctions_match": True}, "ground_truth": "Non-Compliant"},
    {"case": {"correspondent_is_shell_bank": False}, "ground_truth": "Compliant"},
    {"case": {"correspondent_is_shell_bank": True}, "ground_truth": "Non-Compliant"},
    {"case": {"wire_transfer_amount": 75000, "originator_info_present": True}, "ground_truth": "Compliant"},
    {"case": {"wire_transfer_amount": 75000, "originator_info_present": False}, "ground_truth": "Non-Compliant"},
    {"case": {"risk_category": "low", "years_since_update": 5}, "ground_truth": "Compliant"},
    {"case": {"risk_category": "high", "years_since_update": 3}, "ground_truth": "Non-Compliant"},

    # --- Edge cases: missing/incomplete fields, where the rule engine's
    # limitations should actually show up ---
    # Field simply never populated (e.g. extraction step failed to find it).
    # The rule engine will SKIP this rule entirely rather than flag it,
    # which is a real weakness: silence is not the same as compliance.
    {"case": {}, "ground_truth": "Non-Compliant"},

    # Contradictory-looking case: CDD marked complete, but the underlying
    # identity fields that CDD is supposed to guarantee are still False.
    # The rule engine has no way to detect this internal inconsistency
    # since each rule is evaluated independently.
    {"case": {"cdd_completed": True, "aadhaar_present": False, "pan_present": False, "form60_present": False}, "ground_truth": "Non-Compliant"},

    # Boundary value exactly AT the threshold, not above it — tests
    # whether the rule's ">=" vs ">" boundary matches the regulation's
    # actual wording precisely.
    {"case": {"transaction_amount": 50000, "is_walkin": True, "identity_verified": False}, "ground_truth": "Non-Compliant"},
]


def rule_engine_verdict(case: dict, rules: list) -> str:
    """Convert rule engine output into a single Compliant/Non-Compliant verdict."""
    triggered = evaluate_rules(case, rules)
    hard_violations = [r for r in triggered if r["severity"] == "hard"]
    return "Non-Compliant" if hard_violations else "Compliant"


def evaluate_baseline1():
    rules = load_rules()

    y_true = []
    y_pred = []
    results_log = []

    for item in TEST_CASES:
        pred = rule_engine_verdict(item["case"], rules)
        y_true.append(item["ground_truth"])
        y_pred.append(pred)
        results_log.append({
            "case": item["case"],
            "ground_truth": item["ground_truth"],
            "predicted": pred,
            "correct": pred == item["ground_truth"],
        })

    # Metrics
    n = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / n

    # False negatives specifically on hard violations: ground truth says
    # Non-Compliant but rule engine said Compliant (missed a real violation)
    false_negatives = sum(
        1 for t, p in zip(y_true, y_pred)
        if t == "Non-Compliant" and p == "Compliant"
    )
    actual_violations = sum(1 for t in y_true if t == "Non-Compliant")
    false_negative_rate = false_negatives / actual_violations if actual_violations else 0.0

    # Precision/Recall/F1 treating "Non-Compliant" as the positive class
    true_positives = sum(1 for t, p in zip(y_true, y_pred) if t == "Non-Compliant" and p == "Non-Compliant")
    predicted_positives = sum(1 for p in y_pred if p == "Non-Compliant")
    precision = true_positives / predicted_positives if predicted_positives else 0.0
    recall = true_positives / actual_violations if actual_violations else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    summary = {
        "n_cases": n,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_negative_rate_on_violations": round(false_negative_rate, 4),
    }

    return summary, results_log


if __name__ == "__main__":
    summary, results_log = evaluate_baseline1()

    print("=== Baseline 1: Rule-Engine-Only ===\n")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\nMisclassified cases:")
    for r in results_log:
        if not r["correct"]:
            print(f"  {r['case']} -> predicted {r['predicted']}, truth {r['ground_truth']}")

    # Save results for later comparison table
    out_dir = Path(__file__).resolve().parent.parent / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "baseline1_rule_only.json", "w") as f:
        json.dump({"summary": summary, "details": results_log}, f, indent=2, default=str)

    print(f"\nResults saved to eval/results/baseline1_rule_only.json")
