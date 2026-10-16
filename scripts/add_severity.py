"""
Day 1, Step 1: add severity metadata to every rule in rules/rule_base.json.

Adds three fields per rule (idempotent, safe to re-run):
  severity_tier       : critical | major | minor
  severity_score     : S in [0,1], used by the risk score later
  cost_false_approval: relative cost of wrongly approving a violation
                       (cost of a false escalation is fixed at 1)

NOTE: tiers are a DESIGN CHOICE (assumption), not taken from the
regulations. Document the rationale in the paper: critical = sanctions,
terrorism financing, shell banks, anonymous accounts, unreported STRs;
major = core CDD/EDD obligations; minor = periodicity/record-keeping.
"""
import json
from pathlib import Path

RULES_PATH = Path(__file__).resolve().parent.parent / "rules" / "rule_base.json"

TIERS = {
    "critical": {"score": 1.0, "cost": 10,
                 "ids": ["KYC-001", "KYC-007", "KYC-008", "KYC-009", "KYC-013",
                         "FATF-001", "FATF-007", "FATF-008", "FATF-009"]},
    "major":    {"score": 0.6, "cost": 5,
                 "ids": ["KYC-002", "KYC-003", "KYC-004", "KYC-006", "KYC-014",
                         "FATF-002", "FATF-003", "FATF-004", "FATF-005", "FATF-006"]},
    "minor":    {"score": 0.3, "cost": 2,
                 "ids": ["KYC-005", "KYC-010", "KYC-011", "KYC-012"]},
}

lookup = {rid: (tier, v["score"], v["cost"])
          for tier, v in TIERS.items() for rid in v["ids"]}

with open(RULES_PATH, "r", encoding="utf-8") as f:
    rules = json.load(f)

missing = [r["rule_id"] for r in rules if r["rule_id"] not in lookup]
if missing:
    raise SystemExit(f"Rules without a tier assignment: {missing}")

for r in rules:
    tier, score, cost = lookup[r["rule_id"]]
    r["severity_tier"] = tier
    r["severity_score"] = score
    r["cost_false_approval"] = cost

with open(RULES_PATH, "w", encoding="utf-8") as f:
    json.dump(rules, f, indent=2)

from collections import Counter
print(f"Updated {len(rules)} rules:", dict(Counter(r["severity_tier"] for r in rules)))