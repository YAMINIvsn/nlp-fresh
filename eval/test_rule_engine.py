"""
Unit tests for the rule engine, covering one representative test per
rule category (mandatory fields, thresholds, PEP/EDD, STR reporting,
sanctions matching, shell banks, wire transfers, periodic updation,
record retention) so every category is verified independently before
any LLM/ML component is layered on top.

Run with: pytest eval/test_rule_engine.py -v
"""

import sys
from pathlib import Path

# Make the scripts/ folder importable regardless of where pytest is run from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from rule_engine import load_rules, evaluate_rules  # noqa: E402


def get_ids(triggered: list) -> list:
    return [r["rule_id"] for r in triggered]


# ---------- KYC-001 / FATF-001: Anonymous or fictitious accounts ----------

def test_anonymous_account_triggers_violation():
    rules = load_rules()
    case = {"customer_name_verified": False}
    triggered = get_ids(evaluate_rules(case, rules))
    assert "KYC-001" in triggered
    assert "FATF-001" in triggered


def test_verified_name_does_not_trigger():
    rules = load_rules()
    case = {"customer_name_verified": True}
    triggered = get_ids(evaluate_rules(case, rules))
    assert "KYC-001" not in triggered
    assert "FATF-001" not in triggered


# ---------- KYC-003: Walk-in threshold ----------

def test_walkin_above_threshold_without_id_triggers():
    rules = load_rules()
    case = {"transaction_amount": 60000, "is_walkin": True, "identity_verified": False}
    triggered = get_ids(evaluate_rules(case, rules))
    assert "KYC-003" in triggered


def test_walkin_below_threshold_does_not_trigger():
    rules = load_rules()
    case = {"transaction_amount": 20000, "is_walkin": True, "identity_verified": False}
    triggered = get_ids(evaluate_rules(case, rules))
    assert "KYC-003" not in triggered


# ---------- KYC-004: Mandatory identity documents ----------

def test_missing_all_id_documents_triggers():
    rules = load_rules()
    case = {"aadhaar_present": False, "pan_present": False, "form60_present": False}
    triggered = get_ids(evaluate_rules(case, rules))
    assert "KYC-004" in triggered


def test_any_one_id_document_present_does_not_trigger():
    rules = load_rules()
    case = {"aadhaar_present": True, "pan_present": False, "form60_present": False}
    triggered = get_ids(evaluate_rules(case, rules))
    assert "KYC-004" not in triggered


# ---------- KYC-006 / FATF-005: PEP enhanced due diligence ----------

def test_pep_without_senior_approval_triggers():
    rules = load_rules()
    case = {"is_pep": True, "senior_approval": False, "source_of_funds_verified": True}
    triggered = get_ids(evaluate_rules(case, rules))
    assert "KYC-006" in triggered


def test_foreign_pep_with_senior_approval_does_not_trigger_fatf005():
    rules = load_rules()
    case = {"is_foreign_pep": True, "senior_approval": True}
    triggered = get_ids(evaluate_rules(case, rules))
    assert "FATF-005" not in triggered


# ---------- KYC-007 / FATF-008: Suspicious transaction reporting ----------

def test_unreported_suspicious_transaction_triggers():
    rules = load_rules()
    case = {"suspicious_flag": True, "str_filed": False}
    triggered = get_ids(evaluate_rules(case, rules))
    assert "KYC-007" in triggered
    assert "FATF-008" in triggered


def test_reported_suspicious_transaction_does_not_trigger():
    rules = load_rules()
    case = {"suspicious_flag": True, "str_filed": True}
    triggered = get_ids(evaluate_rules(case, rules))
    assert "KYC-007" not in triggered
    assert "FATF-008" not in triggered


# ---------- KYC-008: Sanctions list matching ----------

def test_sanctions_match_triggers_violation():
    rules = load_rules()
    case = {"sanctions_match": True}
    triggered = get_ids(evaluate_rules(case, rules))
    assert "KYC-008" in triggered


# ---------- KYC-013 / FATF-007: Shell bank prohibition ----------

def test_shell_bank_correspondent_triggers():
    rules = load_rules()
    case = {"correspondent_is_shell_bank": True}
    triggered = get_ids(evaluate_rules(case, rules))
    assert "KYC-013" in triggered
    assert "FATF-007" in triggered


# ---------- KYC-014: Wire transfer originator info ----------

def test_large_wire_transfer_missing_originator_triggers():
    rules = load_rules()
    case = {"wire_transfer_amount": 75000, "originator_info_present": False}
    triggered = get_ids(evaluate_rules(case, rules))
    assert "KYC-014" in triggered


def test_small_wire_transfer_missing_originator_does_not_trigger():
    rules = load_rules()
    case = {"wire_transfer_amount": 10000, "originator_info_present": False}
    triggered = get_ids(evaluate_rules(case, rules))
    assert "KYC-014" not in triggered


# ---------- KYC-012: Periodic updation intervals ----------

def test_high_risk_overdue_periodic_updation_triggers():
    rules = load_rules()
    case = {"risk_category": "high", "years_since_update": 3}
    triggered = get_ids(evaluate_rules(case, rules))
    assert "KYC-012" in triggered


def test_low_risk_within_periodic_updation_window_does_not_trigger():
    rules = load_rules()
    case = {"risk_category": "low", "years_since_update": 5}
    triggered = get_ids(evaluate_rules(case, rules))
    assert "KYC-012" not in triggered


# ---------- KYC-010: Record retention ----------

def test_insufficient_record_retention_triggers():
    rules = load_rules()
    case = {"record_retention_years": 3}
    triggered = get_ids(evaluate_rules(case, rules))
    assert "KYC-010" in triggered


# ---------- FATF-009: UNSC terrorism sanctions freezing ----------

def test_unfrozen_unsc_terrorism_match_triggers():
    rules = load_rules()
    case = {"unsc_terrorism_list_match": True, "asset_frozen": False}
    triggered = get_ids(evaluate_rules(case, rules))
    assert "FATF-009" in triggered


# ---------- Clean case: nothing should trigger ----------

def test_fully_compliant_case_triggers_nothing():
    rules = load_rules()
    case = {
        "customer_name_verified": True,
        "cdd_completed": True,
        "identity_verified_independent_source": True,
        "beneficial_owner_identified": True,
        "sanctions_match": False,
        "suspicious_flag": False,
        "str_filed": True,
        "correspondent_is_shell_bank": False,
    }
    triggered = evaluate_rules(case, rules)
    assert triggered == []
