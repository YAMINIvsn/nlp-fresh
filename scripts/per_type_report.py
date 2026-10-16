"""
Per-case-type breakdown of FinBERT-only predictions (uses saved logits).
Shows WHERE the text-only model fails: compliant / violation / missing_info / conflicting.

Run: python scripts/per_type_report.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval" / "results"
LABELS = ["Compliant", "Non-Compliant", "Ambiguous"]


def report(split):
    with open(RESULTS_DIR / f"v2_logits_{split}.json") as f:
        data = json.load(f)
    pred = [int(np.argmax(r["logits"])) for r in data]
    print(f"\n=== {split.upper()} (n={len(data)}) ===")
    by = defaultdict(lambda: [0, 0])
    conf = np.zeros((3, 3), dtype=int)
    for r, p in zip(data, pred):
        by[r["case_type"]][0] += int(p == r["label_id"])
        by[r["case_type"]][1] += 1
        conf[r["label_id"], p] += 1
    print("accuracy by case type:")
    for ct in ["compliant", "violation", "missing_info", "conflicting"]:
        if ct in by:
            c, n = by[ct]
            print(f"  {ct:13s} {c:3d}/{n:3d} = {c/n:.2f}")
    print("confusion matrix (rows = truth, cols = predicted):")
    print(f"  {'':14s}" + "".join(f"{l:>15s}" for l in LABELS))
    for i, l in enumerate(LABELS):
        print(f"  {l:14s}" + "".join(f"{conf[i, j]:15d}" for j in range(3)))


if __name__ == "__main__":
    report("val")
    report("test")