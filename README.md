# NLP-Based Framework for Automated Compliance Auditing of Financial Documents Using Large Language Models

A hybrid compliance-auditing framework that combines language models, machine-learning classification, and rule-based validation for KYC/AML compliance. The project also introduces **DART (Dependency-Aware Regulatory Transition)**, a prototype for analysing regulatory changes and identifying compliance cases that may need to be reviewed again.

## 1. Project Overview

Financial institutions must check documents, customer information, and internal procedures against regulatory requirements such as Know Your Customer (KYC) and Anti-Money Laundering (AML) rules. Manual auditing can be time-consuming, while language models alone may misunderstand regulations or generate unsupported conclusions.

This project explores a hybrid approach that combines machine learning with explicit rules and human review.

The framework has two main components:

1. **Compliance auditing:** Uses a classifier, rule-based checks, and sanctions-name matching to help identify potential compliance issues.
2. **DART:** Analyses changes between regulatory provisions, traces connections to compliance controls, and helps prioritise cases for possible re-audit.

DART is intended to support reviewers. It does not replace qualified regulatory interpretation or make final legal decisions independently.

## 2. System Architecture

### 2.1 Hybrid Compliance Auditing

The original compliance-auditing component includes the following elements:

- **Rule engine:** Applies explicit checks defined in the project's rule configuration.
- **FinBERT classifier:** Uses a language model for classifying regulatory or compliance-related text.
- **Sanctions matching:** Helps identify names that may match entries in sanctions datasets.
- **Escalation:** Identifies cases that may need additional human attention.
- **Audit explanations:** Records decisions and applicable rule checks where supported by the implementation.

The research explores different approaches, including rule-only, classifier-only, and hybrid decisions.

A three-tier decision approach is also evaluated: auto-approve, auto-reject, and escalate for human review. Its results must be interpreted using the evaluation procedure and the number of cases covered by each category.

### 2.2 DART: Regulatory Change and Impact Analysis

DART focuses on what happens when a regulatory requirement changes.

Its intended workflow is:

1. **Obligation extraction:** Extracts structured requirements from supplied regulatory text.
2. **Change detection:** Compares old and new requirements to identify possible additions, modifications, removals, unchanged requirements, and uncertain changes.
3. **Scope validation:** Checks whether the people, conditions, or exceptions covered by a requirement may have changed.
4. **Obligation mapping:** Connects extracted requirements to compliance-control identifiers using reviewed mappings.
5. **Dependency analysis:** Follows known connections between requirements and controls.
6. **Impact propagation:** Identifies audit cases connected to potentially affected controls.
7. **Selective re-audit:** Prioritises cases using impact information, risk scores, mandatory-review flags, and uncertainty indicators.
8. **Human validation:** Keeps uncertain interpretations and unresolved mappings available for review.

### 2.3 DART Safety Measures

The prototype includes safeguards intended to reduce unsafe conclusions:

- An obligation missing from an amendment excerpt is not automatically treated as repealed.
- Supporting quotations can be checked against the supplied source text.
- Uncertain changes and possible scope differences are flagged.
- Graph traversal uses obligation-to-control mappings marked as approved.
- Unresolved changes are recorded rather than silently ignored.
- Human validation remains necessary for regulatory interpretation and high-impact decisions.

These safeguards reduce some risks but do not guarantee that every regulatory change or affected case will be identified correctly.

## 3. Repository Structure

The repository contains the original compliance-auditing code and the DART research module.

```text
nlp-fresh/
├── data/
│   ├── dart/
│   │   ├── extracted/
│   │   ├── provisions/
│   │   ├── examples/
│   │   ├── audit_cases.csv
│   │   ├── audit_ground_truth.csv
│   │   ├── impact_demo_input.json
│   │   ├── impact_simulated_approval.json
│   │   ├── end_to_end_output.json
│   │   └── end_to_end_simulation.json
│   └── ... existing compliance datasets
├── rules/
│   └── rule_base.json
├── models/
│   └── ... model files, if available locally
├── scripts/
│   └── ... existing training and evaluation scripts
├── src/
│   └── dart/
│       ├── llm_client.py
│       ├── obligation_extraction.py
│       ├── change_detection.py
│       ├── scope_validation.py
│       ├── dependency_graph.py
│       ├── impact_propagation.py
│       ├── impact_pipeline.py
│       ├── selective_reaudit.py
│       ├── hybrid_selector.py
│       └── end_to_end_pipeline.py
├── tests/
│   ├── test_hybrid_selector.py
│   └── ... other tests
├── eval/
│   └── results/
│       ├── DART_EVALUATION_REPORT.md
│       └── dart_selector_comparison_v2.json
├── requirements.txt
└── README.md
```

This is a simplified overview, not a complete listing of every repository file. The exact files available may vary by branch and development environment.

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main implementation language |
| PyTorch | Machine-learning model support |
| Hugging Face Transformers | FinBERT and transformer-model tools |
| Scikit-learn | Classification and evaluation utilities |
| Pandas and NumPy | Data processing |
| RapidFuzz | Approximate name matching |
| Pytest | Automated testing |
| Groq API | Language-model requests for DART |
| JSON and CSV | Input data, mappings, and saved results |

The packages required to run the project should be taken from the repository's `requirements.txt` and the imports used by the relevant scripts.

## 5. Data and Regulatory Sources

### 5.1 Compliance Auditing

The original framework works with compliance rules, labelled text, and sanctions-related data.

The project references sources such as:

- RBI Master Direction – Know Your Customer (KYC) Direction.
- Financial Action Task Force (FATF) Recommendations.
- Office of Foreign Assets Control (OFAC) sanctions data.
- OpenSanctions datasets.

The availability, version, licence, and permitted use of each dataset should be verified before reproducing an experiment.

### 5.2 DART Regulatory Data

The DART demonstration includes regulatory text files and version-related examples based on RBI KYC material, including files associated with 2016, 2024, and 2025 provisions.

It also includes synthetic audit cases and example mappings used to test impact propagation and case selection.

**Important:** Synthetic cases demonstrate program behaviour. They are not real customer records and do not establish real-world auditing performance. Verify each regulatory document's official status, effective date, and applicable wording against the authoritative source before drawing legal conclusions.

## 6. Installation and Setup

### Prerequisites

- Python version compatible with the dependencies in `requirements.txt`.
- Git.
- A working Groq API key for DART operations that require the language model.
- The relevant model and dataset files for the original compliance-auditing experiments.

### 6.1 Clone the Repository

```bash
git clone https://github.com/YAMINIvsn/nlp-fresh.git
cd nlp-fresh
```

If you already have the repository locally, open that existing folder instead of cloning it again.

### 6.2 Create a Virtual Environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell prevents activation, follow the Python environment instructions appropriate for your computer rather than changing system-wide security settings unnecessarily.

### 6.3 Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If a command reports a missing package, check `requirements.txt` and the relevant module before installing additional packages.

### 6.4 Configure Groq

Create a `.env` file in the repository root:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
GROQ_TIMEOUT=120
```

Replace the placeholder with your own API key locally. Never commit the real key to GitHub.

The DART LLM client must load these variables for the configured model requests to work. The exact supported settings depend on the current implementation of `src/dart/llm_client.py`.

## 7. Running the Framework

Run commands from the repository root unless stated otherwise. Confirm that the file named in each command exists on your branch.

### 7.1 Test the DART Components

```powershell
python -m pytest tests/ -v
```

This runs the tests under `tests/`. If your project also has tests under `eval/`, run those separately:

```powershell
python -m pytest eval/ -v
```

Tests check the behaviours they cover. Passing tests do not prove that all legal interpretations or real-world cases are correct.

### 7.2 Run Standalone Impact Analysis

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m dart.impact_pipeline --input data/dart/impact_demo_input.json --output data/dart/impact_analysis_output.json
```

This command assumes the DART package is importable from `src` and that the script supports the shown arguments. The program reads the example changed requirements, dependency connections, and audit cases, then saves its results to the specified output file.

### 7.3 Run the End-to-End DART Pipeline

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m dart.end_to_end_pipeline --old data/dart/provisions/rbi_kyc_OLD.txt --new data/dart/provisions/rbi_kyc_2025_new.txt --impact-input data/dart/impact_demo_input.json --old-version 2016 --new-version 2025 --output data/dart/end_to_end_output.json
```

This command assumes the command-line arguments match those supported by the current pipeline. Check the argument definitions in `src/dart/end_to_end_pipeline.py` if your version uses different names.

The intended output contains extracted requirements, change classifications, scope-check results, mapping information, impact analysis, case-selection results, and human-validation indicators.

The end-to-end run requires a working LLM configuration.

### 7.4 Compare DART Selectors

```powershell
$env:PYTHONPATH = "$PWD\src"
python scripts/compare_dart_selectors.py
```

The comparison script uses the configured demonstration input and saves its results to:

```text
eval/results/dart_selector_comparison_v2.json
```

Check the script's input and output paths before running it if you have changed the evaluation files.

## 8. Experimental Evaluation

### 8.1 Original Compliance Auditing

The original project explores rule-based, FinBERT-based, and hybrid compliance decisions.

Report accuracy, precision, recall, and F1 only when the corresponding evaluation outputs can be reproduced from the saved results and evaluation scripts. Include the dataset size, class definitions, and evaluation method when reporting those results.

A high score on automatically resolved cases must not be presented as overall system accuracy unless all cases and the calculation support that conclusion.

### 8.2 DART Selective Re-audit Demonstration

The current selector comparison uses six synthetic cases and a risk threshold of `0.35`.

| Strategy | Selected immediately | Manual triage | Skipped | Immediate review reduction | Graph-affected cases selected immediately |
|---|---:|---:|---:|---:|---:|
| Risk-based baseline | 4 | 0 | 2 | 33.33% | 2 of 3 |
| Impact-first selector | 5 | 0 | 1 | 16.67% | 3 of 3 |
| Hybrid risk-aware selector | 4 | 1 | 1 | 33.33% | 2 of 3 |

These figures describe the saved six-case demonstration. They are not results from a real banking deployment.

**Interpretation:**

- The baseline selects fewer cases immediately but misses one case identified as graph-connected.
- Impact-first selects all three graph-connected cases in this example, at the cost of selecting more cases immediately.
- The hybrid selector places a low-risk affected case into manual triage instead of immediately selecting it or silently ignoring it.
- Manual triage is not the same as completing a re-audit.

The immediate review reduction is calculated relative to reviewing all six cases immediately. It does not measure the final workload after manual triage.

Graph-affected coverage measures coverage of cases identified by the supplied dependency graph. It is not recall against independently labelled regulatory violations.

Detailed results are available in:

```text
eval/results/DART_EVALUATION_REPORT.md
eval/results/dart_selector_comparison_v2.json
```

## 9. Limitations

The current implementation is a research prototype.

- DART's selector demonstration contains only six synthetic cases.
- The supplied connections may not represent every real relationship between regulations, controls, and customer cases.
- AI-generated obligations and change classifications require validation.
- A matching source quotation does not by itself prove that a legal interpretation is correct.
- Obligation-to-control mappings must be reviewed before they are used for graph traversal.
- An excerpt cannot establish that an omitted obligation has been repealed.
- Regulatory publication dates, effective dates, and applicability require verification.
- Risk scores and thresholds need a clear definition and independent evaluation.
- The current demonstration does not establish legal correctness, production readiness, or real-world workload reduction.

## 10. Future Work

Potential improvements include:

- Building a larger dataset of independently reviewed regulatory changes.
- Evaluating obligation extraction and change detection against human-labelled answers.
- Improving the dependency graph and its connection to actual controls.
- Measuring missed affected cases and missed high-severity violations.
- Comparing different risk thresholds and selection strategies.
- Evaluating total reviewer workload, including manual triage.
- Improving regulatory version tracking and effective-date handling.
- Adding clearer explanations and reviewer feedback to the workflow.

## 11. Responsible Use

This project is intended for academic research and decision support. It should not be used as the sole basis for approving or rejecting customers, interpreting legal obligations, or making production compliance decisions.

Reviewers should verify the authoritative regulation, check the evidence, and approve important mappings and decisions before acting on them. API credentials and sensitive customer information must be handled securely.

## 12. Licence and Attribution

Academic research project.

Consult the original RBI, FATF, OFAC, and OpenSanctions sources for authoritative material, applicable terms, and attribution requirements. Do not assume that all referenced datasets share the same licence.
