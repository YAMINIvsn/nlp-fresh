"""
Rule Engine for KYC/AML Compliance Auditing
Evaluates a structured case (dict of extracted document fields) against
the deterministic rule base built from RBI KYC Master Direction and
FATF Recommendations.

NOTE ON SAFETY: This engine uses Python's eval() to evaluate rule
conditions written as plain-English-like boolean expressions in
rule_base.json. This is convenient for a research/academic prototype
where the rule base is authored by the project team, not external
users. It is NOT safe for production use with untrusted rule input.
For a production system, replace this with a proper rules engine
(e.g. `durable_rules`) or hand-written condition-checking functions.
"""

import json
from pathlib import Path

DEFAULT_RULES_PATH = Path(__file__).resolve().parent.parent / "rules" / "rule_base.json"


def load_rules(path: str = None) -> list:
    """Load the JSON rule base from disk."""
    rules_path = Path(path) if path else DEFAULT_RULES_PATH
    with open(rules_path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_rules(case: dict, rules: list) -> list:
    """
    Evaluate a single case (dict of extracted fields) against every rule.

    case: dict of fields extracted from a document/transaction, e.g.
        {
            "customer_name_verified": True,
            "cdd_completed": True,
            "transaction_amount": 60000,
            "is_walkin": True,
            "identity_verified": False,
            ...
        }

    Returns: list of dicts, one per triggered rule, each containing
        rule_id, description, verdict, severity, and source.
    Rules referencing a field that's missing from `case` are silently
    skipped (not triggered) rather than raising an error, since a given
    case will rarely populate every possible field across the whole
    rule base (e.g. a savings-account case won't have wire-transfer fields).
    """
    triggered = []
    for rule in rules:
        try:
            # Restrict eval's builtins for safety; only the case dict's
            # keys are available as variables inside the condition string.
            if eval(rule["condition"], {"__builtins__": {}}, case):
                triggered.append({
                    "rule_id": rule["rule_id"],
                    "description": rule["description"],
                    "verdict": rule["verdict_if_true"],
                    "severity": rule["severity"],
                    "source": rule["source"],
                })
        except NameError:
            # A field referenced in this rule's condition isn't present
            # in this case — the rule doesn't apply here, skip it.
            continue
        except Exception as e:
            # Any other evaluation error (bad syntax, type mismatch, etc.)
            # gets surfaced clearly rather than silently swallowed.
            print(f"Warning: rule {rule['rule_id']} failed to evaluate: {e}")
            continue
    return triggered


def summarize(case: dict, rules: list) -> dict:
    """
    Convenience wrapper: evaluate a case and return a compact summary
    suitable for logging or feeding into the decision-integration layer.
    """
    triggered = evaluate_rules(case, rules)
    hard_violations = [r for r in triggered if r["severity"] == "hard"]
    return {
        "verdict": "Violation" if hard_violations else "Pass",
        "triggered_rules": triggered,
        "num_triggered": len(triggered),
    }


if __name__ == "__main__":
    rules = load_rules()
    print(f"Loaded {len(rules)} rules from rule base.\n")

    # Example case: a walk-in customer above the reporting threshold,
    # with no identity verification and no CDD completed — should
    # trigger multiple hard violations.
    test_case = {
        "customer_name_verified": True,
        "cdd_completed": False,
        "transaction_amount": 60000,
        "is_walkin": True,
        "identity_verified": False,
        "aadhaar_present": False,
        "pan_present": False,
        "form60_present": False,
        "risk_category": None,
        "sanctions_match": False,
    }

    result = summarize(test_case, rules)
    print(f"Verdict: {result['verdict']}")
    print(f"Rules triggered: {result['num_triggered']}\n")
    for r in result["triggered_rules"]:
        print(f"  [{r['rule_id']}] {r['description']}")
        print(f"      -> {r['verdict']} (source: {r['source']})\n")
