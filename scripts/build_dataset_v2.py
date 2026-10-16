"""
Day 1, Step 2: build the v2 benchmark (data/v2/).

Every record has BOTH text (for FinBERT) and structured fields (for the
rule engine), plus metadata for the risk-aware experiments.

Label semantics (document these in the paper):
  Compliant     : relevant fields present, no rule fires.
  Non-Compliant : relevant fields present and explicit, >=1 rule fires,
                  text is consistent with the fields.
  Ambiguous     : evidence is insufficient (field missing -> "missing_info")
                  or contradictory (text claims compliance while the fields
                  show a violation -> "conflicting"). Ground truth = a human
                  must review (should_escalate = True).
Labels are DERIVED FROM THE RULE ENGINE and validated below: the script
aborts if any record's rule outcome disagrees with its intended variant.

Split: group-aware. All instances of one template share a cluster_id and
stay in ONE split (so paraphrase/slot-fill variants never leak across
train/val/test). Stratified by variant, 60/20/20.

Run: python scripts/build_dataset_v2.py
"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rule_engine import load_rules, evaluate_rules  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "v2"
SEED = 42
N_PER_TEMPLATE = 3

ENT = ["an individual customer", "a proprietary firm", "a private limited company",
       "a partnership firm", "a trust"]
BR = ["the Chennai branch", "the Pune branch", "the Kochi branch",
      "the Jaipur branch", "the Lucknow branch", "the Nagpur branch"]

VARIANTS = ["compliant", "violation", "missing_info", "conflicting"]
LABEL = {"compliant": "Compliant", "violation": "Non-Compliant",
         "missing_info": "Ambiguous", "conflicting": "Ambiguous"}
RULE_SHOULD_FIRE = {"compliant": False, "violation": True,
                    "missing_info": False, "conflicting": True}


def amt(r, lo=50000, hi=900000):
    return f"{r.randrange(lo, hi, 5000):,}"


# ---------------- scenario generators: (variant, rng) -> (fields, slots) ----------------
def g_identity(v, r):
    if v == "compliant":
        a, p, f = r.choice([(1, 1, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
        names = [n for n, x in zip(["Aadhaar number", "PAN", "Form 60"], (a, p, f)) if x]
        return ({"aadhaar_present": bool(a), "pan_present": bool(p), "form60_present": bool(f)},
                {"docs": " and ".join(names)})
    if v in ("violation", "conflicting"):
        return {"aadhaar_present": False, "pan_present": False, "form60_present": False}, {}
    return {}, {}


def g_str(v, r):
    s = {"amt": amt(r, 100000, 900000)}
    if v == "compliant":
        return {"suspicious_flag": True, "str_filed": True}, s
    if v in ("violation", "conflicting"):
        return {"suspicious_flag": True, "str_filed": False}, s
    return {"suspicious_flag": True}, s


def g_pep(v, r):
    if v == "compliant":
        return {"is_pep": True, "senior_approval": True, "source_of_funds_verified": True}, {}
    if v == "violation":
        return {"is_pep": True, "senior_approval": False,
                "source_of_funds_verified": r.choice([True, False])}, {}
    if v == "conflicting":
        return {"is_pep": True, "senior_approval": False, "source_of_funds_verified": True}, {}
    return {"is_pep": True}, {}


def g_risk(v, r):
    risk = r.choice(["low", "medium", "high"])
    if v == "compliant":
        return {"risk_category": risk}, {"risk": risk}
    if v == "violation":
        return {"risk_category": None}, {}
    if v == "conflicting":
        return {"risk_category": None}, {"risk": risk}
    return {}, {}


LIM = {"high": 2, "medium": 8, "low": 10}


def g_periodic(v, r):
    risk = r.choice(list(LIM))
    lim = LIM[risk]
    if v == "compliant":
        y = r.randint(1, lim)
    elif v in ("violation", "conflicting"):
        y = lim + r.randint(1, 4)
    else:
        return {"risk_category": risk}, {"risk": risk}
    return {"risk_category": risk, "years_since_update": y}, {"risk": risk, "yrs": y}


def g_wire(v, r):
    s = {"amt": amt(r)}
    if v == "compliant":
        return {"wire_transfer_amount": 1, "originator_info_present": True}, s  # amount patched below
    if v in ("violation", "conflicting"):
        return {"wire_transfer_amount": 1, "originator_info_present": False}, s
    return {"wire_transfer_amount": 1}, s


def g_corr(v, r):
    base = {"is_correspondent_banking": True, "senior_approval": True}
    if v == "compliant":
        return {**base, "correspondent_is_shell_bank": False}, {}
    if v in ("violation", "conflicting"):
        return {**base, "correspondent_is_shell_bank": True}, {}
    return base, {}


def g_sanc(v, r):
    if v == "compliant":
        return {"sanctions_match": False}, {}
    if v in ("violation", "conflicting"):
        return {"sanctions_match": True}, {}
    return {}, {}


def g_walkin(v, r):
    a = r.choice([50000, int(amt(r).replace(",", ""))])
    s = {"amt": f"{a:,}"}
    f = {"transaction_amount": a, "is_walkin": True}
    if v == "compliant":
        return {**f, "identity_verified": True}, s
    if v in ("violation", "conflicting"):
        return {**f, "identity_verified": False}, s
    return f, s


def g_bo(v, r):
    if v == "compliant":
        return {"beneficial_owner_identified": True}, {}
    if v in ("violation", "conflicting"):
        return {"beneficial_owner_identified": False}, {}
    return {}, {}


def g_ret(v, r):
    if v == "compliant":
        y = r.randint(5, 10)
        return {"record_retention_years": y}, {"yrs": y}
    if v in ("violation", "conflicting"):
        y = r.randint(1, 4)
        return {"record_retention_years": y}, {"yrs": y}
    return {}, {}


# ---------------- templates ----------------
SCENARIOS = {
 "identity": dict(topic="customer identity verification", gen=g_identity, targets={"KYC-004"}, T={
  "compliant": [
   "{br} opened the account for {ent} after collecting the {docs}, as required for customer due diligence.",
   "For {ent} onboarded at {br}, the {docs} was obtained and recorded in the KYC file before activation.",
   "KYC file review at {br}: identification of {ent} is complete, supported by the {docs}.",
   "The audit sample shows {ent} was identified through the {docs}; no gaps were noted."],
  "violation": [
   "{br} opened an account for {ent} without obtaining an Aadhaar number, PAN or Form 60.",
   "The KYC file of {ent} at {br} contains no Aadhaar, PAN or Form 60, yet the account is active.",
   "During onboarding of {ent}, {br} accepted the application with no identity number of any kind on record.",
   "Audit found that no Aadhaar, PAN or Form 60 was collected for {ent} onboarded at {br}."],
  "missing_info": [
   "The onboarding record for {ent} at {br} does not state which identity documents were collected.",
   "{br} noted that the identification of {ent} was 'in order' but the file lists no document.",
   "Identity checks for {ent} were reportedly done at {br}; supporting document details are not recorded."],
  "conflicting": [
   "{br} confirmed that {ent} was fully identified using Aadhaar and PAN at onboarding.",
   "The compliance note states identity verification of {ent} at {br} is complete and satisfactory.",
   "Onboarding summary for {ent}: KYC documents verified and filed by {br}."]}),
 "str": dict(topic="suspicious transaction reporting", gen=g_str, targets={"KYC-007", "FATF-008"}, T={
  "compliant": [
   "{br} detected structured cash deposits of about INR {amt} in the account of {ent} and filed an STR with FIU-IND without delay.",
   "An alert on unusual turnover in the account of {ent} was escalated and a suspicious transaction report was submitted to FIU-IND by {br}.",
   "The principal officer filed an STR on the flagged transactions of {ent} at {br} within the prescribed time.",
   "Transactions of {ent} lacking economic rationale were reported to FIU-IND as suspicious by {br}."],
  "violation": [
   "{br} flagged transactions of INR {amt} by {ent} as suspicious but no STR was filed with FIU-IND.",
   "A suspicious alert on {ent} was closed at {br} without any report to the Financial Intelligence Unit.",
   "The STR on the structured deposits of {ent} was drafted by {br} but never submitted.",
   "Despite a documented suspicion, {br} took no reporting action on the transactions of {ent} worth INR {amt}."],
  "missing_info": [
   "{br} flagged unusual activity in the account of {ent}; the file does not show what action followed.",
   "An internal alert was raised on {ent} at {br}; its outcome is not documented.",
   "Unusual cash activity by {ent} was noted at {br} and referred for review, with no record of the decision."],
  "conflicting": [
   "{br} reported that the STR on {ent} was filed with FIU-IND on time.",
   "Compliance sign-off for {ent} states all suspicious transaction reporting obligations were met by {br}.",
   "The monthly return from {br} records the suspicious transaction on {ent} as reported to FIU-IND."]}),
 "pep": dict(topic="politically exposed person handling", gen=g_pep, targets={"KYC-006"}, T={
  "compliant": [
   "{ent} was identified as a politically exposed person; {br} obtained senior management approval and verified the source of funds before onboarding.",
   "For the PEP relationship with {ent}, senior approval was documented and enhanced monitoring started at {br}.",
   "{br} escalated the PEP customer {ent} to senior management, which approved the account after source-of-funds checks.",
   "Enhanced due diligence on PEP {ent} at {br} was completed with senior-level sign-off."],
  "violation": [
   "{br} opened an account for {ent}, a known politically exposed person, without senior management approval.",
   "The relationship manager at {br} onboarded PEP {ent} on their own authority; no senior approval exists.",
   "No senior-level approval was sought before {br} accepted {ent}, who holds a prominent public office.",
   "{ent} was flagged as a PEP but {br} activated the account without escalating for approval."],
  "missing_info": [
   "{ent} was flagged as a PEP at {br}; the file does not show who approved the relationship.",
   "{br} records a PEP screening hit for {ent} and that the account was opened; approval details are absent.",
   "The PEP file for {ent} at {br} is incomplete regarding approval and source of funds."],
  "conflicting": [
   "{br} states that senior management approved the PEP account of {ent} before activation.",
   "The compliance register shows senior sign-off for PEP {ent} at {br}.",
   "Account opening note: PEP status of {ent} handled per policy with approval from senior management at {br}."]}),
 "risk": dict(topic="risk categorization", gen=g_risk, targets={"KYC-005"}, T={
  "compliant": [
   "{ent} was categorised as {risk} risk at {br} based on business activity, geography and transaction profile.",
   "The risk category of {ent} is recorded as {risk} in the core banking system at {br}.",
   "{br} assigned a {risk}-risk rating to {ent} after assessment, as documented in the KYC policy.",
   "Risk categorisation of {ent} at {br} was completed and set to {risk}."],
  "violation": [
   "No risk category has been assigned to {ent} at {br}; the system field is blank.",
   "{br} onboarded {ent} but never categorised the customer as low, medium or high risk.",
   "The risk rating field for {ent} remains empty months after account opening at {br}.",
   "Risk categorisation of {ent} was skipped at {br}."],
  "missing_info": [
   "The file for {ent} at {br} refers to a risk assessment but does not state the resulting category.",
   "{br} mentions that {ent} was 'assessed' without recording a risk level.",
   "A risk review of {ent} took place at {br}; the file does not show its outcome."],
  "conflicting": [
   "{br} confirms {ent} has been categorised as {risk} risk.",
   "According to the compliance note, {ent} carries a {risk}-risk rating at {br}.",
   "The customer profile for {ent} at {br} lists the risk level as {risk}."]}),
 "periodic": dict(topic="periodic KYC updation", gen=g_periodic, targets={"KYC-012"}, T={
  "compliant": [
   "KYC of {ent} ({risk} risk) at {br} was last updated {yrs} years ago, within the prescribed cycle.",
   "{br} refreshed identity and address proofs of the {risk}-risk customer {ent} {yrs} years back, on schedule.",
   "Periodic updation for {ent} at {br} is current; the last review was {yrs} years ago.",
   "The {risk}-risk account of {ent} at {br} is within its updation window, last reviewed {yrs} years ago."],
  "violation": [
   "The {risk}-risk customer {ent} at {br} has not had KYC updated for {yrs} years, beyond the prescribed cycle.",
   "{br} last refreshed KYC of {ent} ({risk} risk) {yrs} years ago; the updation is overdue.",
   "Periodic updation of {ent} at {br} is overdue; the last review was {yrs} years ago.",
   "No KYC refresh has been done for {yrs} years for {ent}, a {risk}-risk customer at {br}."],
  "missing_info": [
   "{br} lists {ent} as a {risk}-risk customer but does not record when KYC was last updated.",
   "The date of the last periodic review of {ent} ({risk} risk) at {br} is not on file.",
   "Updation status for {ent} at {br} is marked 'in progress' without a start date."],
  "conflicting": [
   "{br} confirms KYC of the {risk}-risk customer {ent} was updated recently and is fully current.",
   "Compliance note: periodic updation for {ent} at {br} is up to date.",
   "{ent} ({risk} risk) at {br} has a current, completed KYC refresh according to the branch."]}),
 "wire": dict(topic="wire transfer originator information", gen=g_wire, targets={"KYC-014"}, T={
  "compliant": [
   "{br} processed a wire transfer of INR {amt} for {ent} with the originator's name, address and account number attached.",
   "The INR {amt} transfer by {ent} from {br} carried complete originator information.",
   "Originator details were captured in full for the INR {amt} wire of {ent} at {br}.",
   "{br} verified that the INR {amt} remittance of {ent} included all required originator fields."],
  "violation": [
   "{br} sent a wire transfer of INR {amt} for {ent} without any originator information.",
   "The INR {amt} transfer by {ent} left {br} with the originator name and address missing.",
   "A domestic wire of INR {amt} for {ent} was processed at {br} lacking originator details.",
   "Originator information was not attached to the INR {amt} remittance of {ent} from {br}."],
  "missing_info": [
   "A wire transfer of INR {amt} by {ent} was processed at {br}; the record does not show which originator fields were captured.",
   "{br} refers to 'standard originator details' for the INR {amt} transfer of {ent} without listing them.",
   "The INR {amt} remittance of {ent} at {br} has a reference number, but the originator fields are not itemised."],
  "conflicting": [
   "{br} confirms the INR {amt} wire of {ent} carried full originator name, address and account number.",
   "Audit note: all originator information was present on the INR {amt} transfer by {ent} from {br}.",
   "The INR {amt} remittance of {ent} at {br} is recorded as fully compliant on originator data."]}),
 "corr": dict(topic="correspondent banking due diligence", gen=g_corr, targets={"KYC-013", "FATF-007"}, T={
  "compliant": [
   "Before opening the correspondent relationship with a respondent bank, {br} confirmed it has a physical presence and obtained senior management approval.",
   "The correspondent banking file at {br} shows the respondent is a regulated, affiliated bank and the relationship was approved at senior level.",
   "{br} completed due diligence on the respondent bank, found no shell-bank characteristics and recorded senior approval.",
   "Correspondent arrangement reviewed at {br}: the respondent bank has a physical presence in its country of incorporation, and approval is on file."],
  "violation": [
   "{br} set up a correspondent relationship with a bank that has no physical presence and no regulated group affiliation.",
   "The respondent bank in the correspondent arrangement at {br} was found to be a shell bank.",
   "A shell bank, incorporated where it has no physical presence, was accepted as a correspondent by {br}.",
   "{br} continues a correspondent account for an unaffiliated bank without any physical presence in its country."],
  "missing_info": [
   "The correspondent file at {br} describes the respondent bank's business but does not state whether it has a physical presence.",
   "{br} lacks a record of whether the respondent bank is affiliated with a regulated financial group.",
   "Due diligence on the respondent bank at {br} is referenced but its findings on physical presence are not on file."],
  "conflicting": [
   "{br} states that the respondent bank has a verified physical presence and is part of a regulated group.",
   "Compliance confirmed that no shell-bank risk exists in the correspondent relationship handled by {br}.",
   "The respondent bank's physical presence was verified, according to the correspondent banking note at {br}."]}),
 "sanctions": dict(topic="sanctions screening", gen=g_sanc, targets={"KYC-008"}, T={
  "compliant": [
   "{ent} was screened against the sanctions lists at {br} before onboarding and no match was found.",
   "Sanctions screening of {ent} at {br} returned no hit against the lists circulated by RBI.",
   "{br} ran {ent} through the UN and RBI sanctions lists; the result was clear.",
   "No match against any sanctions list was found for {ent} during onboarding at {br}."],
  "violation": [
   "Screening at {br} showed the name of {ent} matches an entry on a sanctions list, yet the account remains open.",
   "{ent} is a confirmed match to a sanctions-listed party and {br} has not blocked the account.",
   "{br} onboarded {ent} even though the name appears on a sanctions list.",
   "A sanctions hit on {ent} was recorded at {br} but no action was taken."],
  "missing_info": [
   "{br} says {ent} was 'checked' before onboarding but the screening result is not recorded.",
   "The file for {ent} at {br} mentions sanctions screening without stating the outcome.",
   "Sanctions screening of {ent} at {br} is referenced, with no evidence of the lists used."],
  "conflicting": [
   "{br} confirms sanctions screening of {ent} was clear with no matches.",
   "According to the onboarding note, {ent} cleared all sanctions checks at {br}.",
   "Compliance sign-off at {br} states there is no sanctions exposure for {ent}."]}),
 "walkin": dict(topic="walk-in customer identification", gen=g_walkin, targets={"KYC-003"}, T={
  "compliant": [
   "{br} collected and verified identity documents from the walk-in customer {ent} before a transaction of INR {amt}.",
   "The walk-in transaction of INR {amt} by {ent} at {br} was preceded by full identity verification.",
   "{ent} walked in at {br} for an INR {amt} transaction and was identified through an officially valid document.",
   "Identity of the walk-in customer {ent} was verified at {br} for the INR {amt} transaction, as the threshold requires."],
  "violation": [
   "{br} carried out a walk-in transaction of INR {amt} for {ent} without verifying identity.",
   "No identification was taken from walk-in customer {ent} for the INR {amt} transaction at {br}.",
   "A walk-in transaction of INR {amt} was processed for {ent} at {br} with no ID check.",
   "{br} handled an INR {amt} cash transaction for {ent}, who was not identified."],
  "missing_info": [
   "A walk-in transaction of INR {amt} by {ent} took place at {br}; the record does not state whether identity was checked.",
   "{br} logged the INR {amt} walk-in transaction of {ent} with no information on identification.",
   "The walk-in file for {ent} at {br} (INR {amt}) is silent on identity verification."],
  "conflicting": [
   "{br} confirms identity was verified for walk-in customer {ent} before the INR {amt} transaction.",
   "The walk-in transaction of INR {amt} by {ent} at {br} is recorded as fully identified.",
   "Branch note: KYC identification completed for {ent} on the INR {amt} walk-in transaction at {br}."]}),
 "bo": dict(topic="beneficial ownership disclosure", gen=g_bo, targets={"FATF-003"}, T={
  "compliant": [
   "{br} identified and verified the beneficial owners of {ent} through registry records before opening the account.",
   "The beneficial owners of {ent} were named and their identities verified at {br}.",
   "For {ent}, {br} traced ownership to the natural persons with controlling interest and verified them.",
   "Beneficial ownership of {ent} was established and documented in the KYC file at {br}."],
  "violation": [
   "{br} opened an account for {ent} without identifying its beneficial owners.",
   "No inquiry into the ownership structure of {ent} was made at {br}.",
   "The beneficial owners of {ent} remain unidentified in the file held at {br}.",
   "{br} accepted {ent} based on the signatory alone, with no beneficial-owner identification."],
  "missing_info": [
   "{br} obtained an ownership declaration from {ent}, but the file does not show whether the owners were identified.",
   "The ownership chart of {ent} at {br} is mentioned but not attached; identification status is unclear.",
   "A change in the shareholding of {ent} was noted at {br} with no update on beneficial-owner identification."],
  "conflicting": [
   "{br} states the beneficial owners of {ent} were identified and verified.",
   "Compliance records at {br} show beneficial ownership of {ent} fully established.",
   "The onboarding checklist for {ent} at {br} ticks beneficial-owner identification as done."]}),
 "retention": dict(topic="record retention", gen=g_ret, targets={"KYC-010"}, T={
  "compliant": [
   "{br} retains transaction and identification records of {ent} for {yrs} years, meeting the five-year minimum.",
   "Records of {ent} at {br} are preserved for {yrs} years after the relationship, as required.",
   "The record-retention schedule at {br} keeps customer records of {ent} for {yrs} years.",
   "Identification and transaction data for {ent} are stored for {yrs} years at {br}."],
  "violation": [
   "{br} destroys transaction records of {ent} after {yrs} years, below the five-year minimum.",
   "The retention policy applied at {br} keeps records of {ent} for only {yrs} years.",
   "Identification records for {ent} were purged at {br} after {yrs} years.",
   "Records of {ent} at {br} are retained for {yrs} years, which falls short of the mandated period."],
  "missing_info": [
   "The retention period for records of {ent} at {br} is not stated in the file.",
   "{br} says records of {ent} are 'kept as per policy' without specifying the duration.",
   "Record-keeping for {ent} at {br} is referenced but the retention period is undocumented."],
  "conflicting": [
   "{br} confirms records of {ent} are retained for at least five years as required.",
   "According to the audit note, the records of {ent} at {br} meet the statutory retention period.",
   "The retention policy at {br} is stated to comply with the five-year requirement for {ent}."]}),
}


def build():
    rng = random.Random(SEED)
    rules = load_rules()
    sev = {r["rule_id"]: r.get("severity_score", 1.0) for r in rules}
    records, errors = [], []

    for sname, sc in SCENARIOS.items():
        for v in VARIANTS:
            for ti, tpl in enumerate(sc["T"][v]):
                cluster = f"{sname}-{v}-{ti}"
                seen, tries = set(), 0
                while len(seen) < N_PER_TEMPLATE and tries < 60:
                    tries += 1
                    fields, slots = sc["gen"](v, rng)
                    s = {"ent": rng.choice(ENT), "br": rng.choice(BR)}
                    s.update(slots)
                    # wire: make amount in fields match the amount in the text
                    if sname == "wire":
                        fields = dict(fields)
                        fields["wire_transfer_amount"] = int(s["amt"].replace(",", ""))
                    if sname == "str":
                        fields = dict(fields)
                    text = tpl.format(**s)
                    if text in seen:
                        continue
                    seen.add(text)
                    fired = evaluate_rules(fields, rules)
                    fired_ids = {x["rule_id"] for x in fired}
                    ok = (bool(fired_ids) == RULE_SHOULD_FIRE[v])
                    if RULE_SHOULD_FIRE[v]:
                        ok = ok and bool(fired_ids & sc["targets"])
                    if not ok:
                        errors.append((cluster, fields, sorted(fired_ids)))
                    records.append({
                        "text": text, "fields": fields, "label": LABEL[v],
                        "case_type": v, "topic": sc["topic"], "scenario": sname,
                        "cluster_id": cluster, "rules_triggered": sorted(fired_ids),
                        "max_severity_score": max([sev[i] for i in fired_ids], default=0.0),
                        "should_escalate": LABEL[v] == "Ambiguous",
                    })
                if len(seen) < N_PER_TEMPLATE:
                    print(f"note: only {len(seen)} unique texts for cluster {cluster}")

    if errors:
        for e in errors[:10]:
            print("LABEL/RULE MISMATCH:", e)
        raise SystemExit(f"{len(errors)} records disagree with the rule engine; aborting.")
    return records


def split(records):
    rng = random.Random(SEED)
    by_variant = defaultdict(set)
    for r in records:
        by_variant[r["case_type"]].add(r["cluster_id"])
    assign = {}
    for v, clusters in by_variant.items():
        cl = sorted(clusters)
        rng.shuffle(cl)
        n = len(cl)
        ntr, nva = round(0.6 * n), round(0.2 * n)
        for i, c in enumerate(cl):
            assign[c] = "train" if i < ntr else ("val" if i < ntr + nva else "test")
    out = {"train": [], "val": [], "test": []}
    for r in records:
        out[assign[r["cluster_id"]]].append(r)
    for k in out:
        rng.shuffle(out[k])
        for i, r in enumerate(out[k]):
            r["id"] = f"{k}_{i:04d}"
    return out


if __name__ == "__main__":
    records = build()
    splits = split(records)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "all.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=1)
    for k, d in splits.items():
        with open(OUT_DIR / f"{k}.json", "w", encoding="utf-8") as f:
            json.dump(d, f, indent=1)

    print(f"Total records: {len(records)}  (rule-validated, 0 mismatches)")
    for k, d in splits.items():
        print(f"{k:5s} n={len(d):3d} labels={dict(Counter(x['label'] for x in d))} "
              f"types={dict(Counter(x['case_type'] for x in d))}")
    # leakage checks
    cl = {k: {x["cluster_id"] for x in d} for k, d in splits.items()}
    assert not (cl["train"] & cl["val"] or cl["train"] & cl["test"] or cl["val"] & cl["test"])
    texts = {k: {x["text"] for x in d} for k, d in splits.items()}
    dup = (texts["train"] & texts["val"]) | (texts["train"] & texts["test"]) | (texts["val"] & texts["test"])
    print(f"cluster overlap across splits: 0 | exact-text overlap across splits: {len(dup)}")
    # how often the rule engine alone fires per case type (the silent-pass problem)
    print("rule engine fires on:", {v: f"{sum(1 for r in records if r['case_type']==v and r['rules_triggered'])}/{sum(1 for r in records if r['case_type']==v)}" for v in VARIANTS})