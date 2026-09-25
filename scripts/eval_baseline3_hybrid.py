"""
Baseline 3: Binary-Override Hybrid (Osei-Asiamah style)

Combines the rule engine (Baseline 1) and fine-tuned FinBERT (Baseline 2):
if the rule engine detects a hard violation, its verdict OVERRIDES
FinBERT's prediction, regardless of FinBERT's confidence. FinBERT's
judgment only matters when the rule engine stays silent (no rule fires).

Requires: a fine-tuned FinBERT model already saved at models/finbert_kyc/
(produced by finetune_finbert.py). Run that script first if you haven't.

Each test case below pairs:
  - "text": a clause description (fed to FinBERT)
  - "fields": structured fields extracted from the same underlying case
    (fed to the rule engine) — in a real pipeline, an NER/extraction
    step would produce these fields from the same document the text
    clause came from.
  - "ground_truth": Compliant / Non-Compliant, as a human reviewer
    would judge given the full picture.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rule_engine import load_rules, evaluate_rules

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "finbert_kyc"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval" / "results"

LABEL_MAP = {0: "Compliant", 1: "Non-Compliant", 2: "Ambiguous"}

# Paired test cases: text for FinBERT, structured fields for the rule engine.
# Ground truth reflects the correct real-world verdict for the FULL case.
COMBINED_TEST_CASES = [
    {
        "text": "The account was opened after collecting the customer's Aadhaar number and PAN, both verified against the issuing authority.",
        "fields": {"aadhaar_present": True, "pan_present": True, "form60_present": False},
        "ground_truth": "Compliant",
    },
    {
        "text": "Customer onboarding was completed using a self-attested photograph only; no officially valid document was collected.",
        "fields": {"aadhaar_present": False, "pan_present": False, "form60_present": False},
        "ground_truth": "Non-Compliant",
    },
    {
        "text": "The branch identified an unusually structured series of cash deposits just below the reporting threshold but took no further action.",
        "fields": {"suspicious_flag": True, "str_filed": False},
        "ground_truth": "Non-Compliant",
    },
    {
        "text": "Upon detecting a pattern of transactions inconsistent with declared business activity, an STR was filed within the prescribed timeframe.",
        "fields": {"suspicious_flag": True, "str_filed": True},
        "ground_truth": "Compliant",
    },
    {
        "text": "No risk category was ever assigned to the account, and the field remains blank more than a year after opening.",
        "fields": {"risk_category": None},
        "ground_truth": "Non-Compliant",
    },
    {
        "text": "The customer was categorised as high-risk based on documented parameters, reviewed every six months as required.",
        "fields": {"risk_category": "high"},
        "ground_truth": "Compliant",
    },
    {
        "text": "The correspondent relationship was established with a bank that has no physical presence and no affiliation with any regulated group.",
        "fields": {"correspondent_is_shell_bank": True},
        "ground_truth": "Non-Compliant",
    },
    {
        "text": "The account was marked medium risk at opening, but there is no record of any subsequent periodic review.",
        "fields": {},  # extraction couldn't determine review status -> ambiguous in text, rule engine has nothing to check
        "ground_truth": "Non-Compliant",  # a missing periodic review is itself the violation
    },
    {
        "text": "The customer's identity was verified using an officially valid document, cross-checked against the issuing authority's records.",
        "fields": {"aadhaar_present": True, "pan_present": True, "form60_present": False},
        "ground_truth": "Compliant",
    },
]


def rule_engine_verdict(fields: dict, rules: list) -> tuple:
    """Returns (verdict, triggered_rules)."""
    triggered = evaluate_rules(fields, rules)
    hard_violations = [r for r in triggered if r["severity"] == "hard"]
    verdict = "Non-Compliant" if hard_violations else None  # None = rule engine has no opinion
    return verdict, triggered


def finbert_predict(texts: list, model_dir: Path) -> list:
    """Run the fine-tuned FinBERT model on a list of texts, return predicted labels."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    predictions = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            outputs = model(**inputs)
            pred_id = torch.argmax(outputs.logits, dim=1).item()
            predictions.append(LABEL_MAP[pred_id])

    return predictions


def hybrid_override_verdict(rule_verdict: str, finbert_pred: str) -> str:
    """
    Binary override logic: rule engine wins if it has an opinion.
    FinBERT's prediction only matters when the rule engine is silent.
    FinBERT's "Ambiguous" is treated as "Compliant" for this binary
    comparison, since the hybrid baseline only distinguishes
    Compliant vs Non-Compliant (matching Osei-Asiamah et al.'s design).
    """
    if rule_verdict == "Non-Compliant":
        return "Non-Compliant"
    # Rule engine had no opinion -> defer to FinBERT
    return "Non-Compliant" if finbert_pred == "Non-Compliant" else "Compliant"


def evaluate_baseline3():
    rules = load_rules()
    texts = [c["text"] for c in COMBINED_TEST_CASES]

    print("Running FinBERT predictions...")
    finbert_preds = finbert_predict(texts, MODEL_DIR)

    y_true = []
    y_pred = []
    details = []

    for case, finbert_pred in zip(COMBINED_TEST_CASES, finbert_preds):
        rule_verdict, triggered = rule_engine_verdict(case["fields"], rules)
        final_verdict = hybrid_override_verdict(rule_verdict, finbert_pred)

        y_true.append(case["ground_truth"])
        y_pred.append(final_verdict)
        details.append({
            "text": case["text"],
            "fields": case["fields"],
            "rule_verdict": rule_verdict,
            "finbert_pred": finbert_pred,
            "final_verdict": final_verdict,
            "ground_truth": case["ground_truth"],
            "correct": final_verdict == case["ground_truth"],
            "overridden_by_rule": rule_verdict == "Non-Compliant",
        })

    n = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / n

    true_positives = sum(1 for t, p in zip(y_true, y_pred) if t == "Non-Compliant" and p == "Non-Compliant")
    predicted_positives = sum(1 for p in y_pred if p == "Non-Compliant")
    actual_positives = sum(1 for t in y_true if t == "Non-Compliant")
    precision = true_positives / predicted_positives if predicted_positives else 0.0
    recall = true_positives / actual_positives if actual_positives else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    false_negatives = sum(1 for t, p in zip(y_true, y_pred) if t == "Non-Compliant" and p == "Compliant")
    fn_rate = false_negatives / actual_positives if actual_positives else 0.0

    summary = {
        "n_cases": n,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_negative_rate_on_violations": round(fn_rate, 4),
        "n_overridden_by_rule": sum(1 for d in details if d["overridden_by_rule"]),
    }

    return summary, details


if __name__ == "__main__":
    summary, details = evaluate_baseline3()

    print("\n=== Baseline 3: Binary-Override Hybrid ===\n")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\nCase-by-case breakdown:")
    for d in details:
        flag = "✓" if d["correct"] else "✗"
        override_note = " [RULE OVERRIDE]" if d["overridden_by_rule"] else " [deferred to FinBERT]"
        print(f"  {flag} rule={d['rule_verdict']} finbert={d['finbert_pred']} "
              f"-> final={d['final_verdict']} (truth={d['ground_truth']}){override_note}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "baseline3_hybrid_override.json", "w") as f:
        json.dump({"summary": summary, "details": details}, f, indent=2, default=str)
    print(f"\nResults saved to eval/results/baseline3_hybrid_override.json")
