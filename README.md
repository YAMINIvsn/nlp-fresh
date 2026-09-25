# NLP-Based Framework for Automated Compliance Auditing of Financial Documents

A hybrid LLM + rule-engine framework for automating KYC/AML compliance auditing, combining deterministic regulatory rule-checking with a fine-tuned language model and a confidence-calibrated escalation layer.

## Problem

Financial institutions must audit vast volumes of unstructured documents against frameworks like KYC and AML. Manual auditing doesn't scale, while pure LLM-based approaches lack the rigor to enforce non-negotiable regulatory requirements. This project proposes and evaluates a hybrid architecture that combines LLM semantic understanding with deterministic rule validation, and introduces a novel three-tier escalation mechanism that improves on the binary rule-override designs used in prior work.

## Architecture

```
Input case (text clause + structured fields)
        |
        +---> Rule engine (23 rules from RBI KYC Direction + FATF Recommendations)
        |
        +---> FinBERT classifier (fine-tuned, outputs label + confidence)
        |
        v
Escalation decision (confidence threshold: 0.70)
        |
        +---> Auto-approve   (rule + FinBERT agree, compliant)
        +---> Auto-reject    (hard rule violation detected)
        +---> Escalate       (rule silent, FinBERT uncertain or ambiguous)
        |
        v
Explainable audit report (SHAP + rule trace log)
```

Unlike prior binary-override hybrids (rule engine always wins, LLM decides otherwise), this model treats "the system doesn't know" as a first-class outcome, routing genuinely uncertain cases to a human reviewer instead of silently defaulting to a guess.

## Repository structure

```
├── data/
│   ├── raw/
│   │   └── sanctions/          # OFAC SDN.CSV, ADD.CSV, ALT.CSV, OpenSanctions targets.simple.csv
│   └── processed/
│       ├── kyc_clauses.json    # 72 labeled synthetic KYC/AML clauses
│       ├── train.json / val.json / test.json
│       └── sanctions_entities.json
├── rules/
│   └── rule_base.json          # 23 rules extracted from RBI KYC Direction + FATF Recommendations
├── scripts/
│   ├── rule_engine.py
│   ├── build_sanctions_db.py
│   ├── entity_matching.py
│   ├── split_dataset.py
│   ├── finetune_finbert.py
│   ├── eval_baseline1_rule_only.py
│   ├── eval_baseline3_hybrid.py
│   └── eval_escalation_model.py
├── eval/
│   ├── test_rule_engine.py     # pytest unit tests for the rule engine
│   └── results/                # JSON outputs from each baseline evaluation
├── models/
│   └── finbert_kyc/             # fine-tuned model weights (gitignored — see below)
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Data sources

- RBI Master Direction – Know Your Customer (KYC) Direction, 2016: https://rbi.org.in/Scripts/BS_ViewMasDirections.aspx?id=10292
- FATF Recommendations: https://www.fatf-gafi.org/en/publications/Fatfrecommendations/Fatf-recommendations.html
- OFAC SDN List: https://sanctionslistservice.ofac.treas.gov/
- OpenSanctions: https://www.opensanctions.org/datasets/ (bulk download, non-commercial use)

Place downloaded files in `data/raw/sanctions/` as described in each script's docstring.

## Running the pipeline

```bash
# 1. Rule engine unit tests
pytest eval/test_rule_engine.py -v

# 2. Build the sanctions entity database
python scripts/build_sanctions_db.py

# 3. Split the clause dataset
python scripts/split_dataset.py

# 4. Baseline 1: rule-engine-only
python scripts/eval_baseline1_rule_only.py

# 5. Baseline 2: fine-tune FinBERT
python scripts/finetune_finbert.py

# 6. Baseline 3: binary-override hybrid
python scripts/eval_baseline3_hybrid.py

# 7. Novel model: three-tier escalation
python scripts/eval_escalation_model.py
```

## Results so far

| System | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline 1: Rule-engine-only | 0.957 | 1.0 | 0.923 | 0.960 |
| Baseline 2: FinBERT-only | 0.667 | 0.756 | 0.667 | 0.639 |
| Baseline 3: Binary-override hybrid | 0.889 | 1.0 | 0.8 | 0.889 |
| Novel: Three-tier escalation | 1.0 (auto-resolved) | — | — | — |

Full details, including case-by-case breakdowns, are in `eval/results/`.

## Known limitations

- The synthetic clause dataset (72 examples) is LLM-generated, not sourced from real audit documents — a known constraint across this research area due to data confidentiality.
- Test sets are small (9–23 cases); results should be read as proof-of-concept, not statistically powered benchmarks.
- The rule engine and FinBERT operate on different input types (structured fields vs. text); a real deployment would need an NER/field-extraction layer between raw documents and both branches, which is out of scope for this project.
- The 0.70 confidence threshold for escalation was not tuned via ablation.

## License

Academic project. Regulatory source documents (RBI, FATF) are publicly available government/intergovernmental publications, not redistributed here — only derived rule summaries with citations are included.
