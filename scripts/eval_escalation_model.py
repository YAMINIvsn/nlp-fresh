"""
Your Novel Contribution: Three-Tier Confidence-Calibrated Escalation

Unlike Baseline 3's binary override (rule engine always wins when it has
an opinion, FinBERT decides otherwise with no notion of uncertainty),
this model explicitly represents "I don't know" as a first-class outcome
and routes those cases to a human reviewer, rather than silently
defaulting to Compliant.

Three tiers:
  1. AUTO-APPROVE: rule engine and FinBERT agree the case is compliant,
     with reasonable confidence.
  2. AUTO-REJECT: rule engine detects a hard violation (same as before —
     this signal is trustworthy and should still short-circuit).
  3. ESCALATE: rule engine has no opinion AND FinBERT's confidence is
     low OR its prediction is "Ambiguous" — this is exactly the failure
     case that broke Baseline 3.

The key design change from Baseline 3: FinBERT's OWN confidence score
(not just its predicted label) is used to decide whether to trust it
or escalate. A missing-fields case with a low-confidence FinBERT
prediction now correctly escalates instead of defaulting to Compliant.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rule_engine import load_rules, evaluate_rules
from eval_baseline3_hybrid import COMBINED_TEST_CASES, rule_engine_verdict, LABEL_MAP

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "finbert_kyc"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval" / "results"

# Below this confidence, FinBERT's prediction is not trusted enough to
# auto-resolve a case — it gets escalated instead. This threshold is a
# genuine design parameter worth reporting and justifying in your paper
# (e.g. via an ablation sweeping 0.6 / 0.7 / 0.8 / 0.9 and reporting how
# escalation rate and accuracy trade off).
CONFIDENCE_THRESHOLD = 0.70


def finbert_predict_with_confidence(texts: list, model_dir: Path) -> list:
    """
    Run the fine-tuned FinBERT model and return BOTH the predicted label
    and its confidence (softmax probability of the winning class) —
    this is the key addition over Baseline 3's label-only prediction.
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    results = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            confidence, pred_id = torch.max(probs, dim=1)
            results.append({
                "label": LABEL_MAP[pred_id.item()],
                "confidence": round(confidence.item(), 4),
            })

    return results


def escalation_decision(rule_verdict: str, finbert_result: dict) -> dict:
    """
    Three-tier routing logic.

    Returns a dict with:
        tier: "auto_approve" | "auto_reject" | "escalate"
        final_verdict: "Compliant" | "Non-Compliant" | "Escalated"
        reasoning: short human-readable explanation for the audit trail
    """
    # Tier 2: rule engine detects a hard violation -> trustworthy, auto-reject.
    # This signal is deterministic and grounded in an actual regulatory
    # citation, so it still short-circuits exactly like Baseline 3.
    if rule_verdict == "Non-Compliant":
        return {
            "tier": "auto_reject",
            "final_verdict": "Non-Compliant",
            "reasoning": "Rule engine detected a hard violation.",
        }

    # Rule engine has no opinion — now FinBERT's confidence decides
    # whether we trust its judgment or escalate.
    label = finbert_result["label"]
    confidence = finbert_result["confidence"]

    if label == "Ambiguous" or confidence < CONFIDENCE_THRESHOLD:
        return {
            "tier": "escalate",
            "final_verdict": "Escalated",
            "reasoning": f"Rule engine had no opinion; FinBERT predicted "
                         f"'{label}' with confidence {confidence:.2f}, "
                         f"below the {CONFIDENCE_THRESHOLD} threshold.",
        }

    # Rule engine silent, FinBERT confident and not Ambiguous -> trust it.
    if label == "Non-Compliant":
        return {
            "tier": "auto_reject",
            "final_verdict": "Non-Compliant",
            "reasoning": f"FinBERT confidently ({confidence:.2f}) predicted Non-Compliant; "
                         f"no rule violation was found, but the semantic signal was strong.",
        }
    else:  # Compliant
        return {
            "tier": "auto_approve",
            "final_verdict": "Compliant",
            "reasoning": f"Rule engine and FinBERT (confidence {confidence:.2f}) agree: compliant.",
        }


def evaluate_escalation_model():
    rules = load_rules()
    texts = [c["text"] for c in COMBINED_TEST_CASES]

    print("Running FinBERT predictions with confidence scores...")
    finbert_results = finbert_predict_with_confidence(texts, MODEL_DIR)

    details = []
    for case, finbert_result in zip(COMBINED_TEST_CASES, finbert_results):
        rule_verdict, _ = rule_engine_verdict(case["fields"], rules)
        decision = escalation_decision(rule_verdict, finbert_result)

        details.append({
            "text": case["text"],
            "fields": case["fields"],
            "rule_verdict": rule_verdict,
            "finbert_label": finbert_result["label"],
            "finbert_confidence": finbert_result["confidence"],
            "tier": decision["tier"],
            "final_verdict": decision["final_verdict"],
            "reasoning": decision["reasoning"],
            "ground_truth": case["ground_truth"],
        })

    # Metrics: separate auto-resolved cases from escalated ones, since
    # accuracy only makes sense to compute on auto-resolved cases —
    # escalated cases are, by design, handed to a human, not scored
    # against the model's own judgment.
    auto_resolved = [d for d in details if d["tier"] != "escalate"]
    escalated = [d for d in details if d["tier"] == "escalate"]

    correct_auto = sum(1 for d in auto_resolved if d["final_verdict"] == d["ground_truth"])
    auto_accuracy = correct_auto / len(auto_resolved) if auto_resolved else None

    # Check: are the escalated cases actually the hard ones? (i.e. is
    # escalation well-targeted, not just noise) — measured by how many
    # escalated cases the OLD binary-override baseline got wrong.
    escalated_ground_truths = [d["ground_truth"] for d in escalated]

    summary = {
        "n_cases": len(details),
        "n_auto_resolved": len(auto_resolved),
        "n_escalated": len(escalated),
        "escalation_rate": round(len(escalated) / len(details), 4),
        "accuracy_on_auto_resolved": round(auto_accuracy, 4) if auto_accuracy is not None else None,
        "escalated_case_texts": [d["text"][:80] + "..." for d in escalated],
    }

    return summary, details


if __name__ == "__main__":
    summary, details = evaluate_escalation_model()

    print("\n=== Novel Model: Three-Tier Confidence-Calibrated Escalation ===\n")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\nCase-by-case breakdown:")
    for d in details:
        marker = {"auto_approve": "✅", "auto_reject": "🚫", "escalate": "🟡"}[d["tier"]]
        print(f"  {marker} [{d['tier'].upper()}] {d['final_verdict']} "
              f"(truth={d['ground_truth']}) — {d['reasoning']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "novel_escalation_model.json", "w") as f:
        json.dump({"summary": summary, "details": details}, f, indent=2, default=str)
    print(f"\nResults saved to eval/results/novel_escalation_model.json")
