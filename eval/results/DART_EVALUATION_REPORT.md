# DART Selective Re-Audit Evaluation

## Evaluation setup

This is a six-case synthetic demonstration. Graph-affected coverage
measures coverage of cases identified by the supplied dependency graph;
it is not recall against independently labelled regulatory violations.

Risk threshold: 0.35

## Strategy comparison

| Strategy | Selected | Manual triage | Skipped | Re-audit reduction | Affected-case re-audit coverage |
|---|---:|---:|---:|---:|---:|
| Risk-based baseline | 4 | 0 | 2 | 33.33% | 66.67% |
| Impact-first selector | 5 | 0 | 1 | 16.67% | 100.00% |
| Hybrid risk-aware selector | 4 | 1 | 1 | 16.67% | 66.67% |

## Hybrid case decisions

- Re-audit: CASE-001, CASE-002, CASE-004, CASE-005
- Manual triage: CASE-006
- Skipped: CASE-003

## Interpretation

- The risk-based baseline skips CASE-006, which the dependency graph
  identifies as potentially affected.
- Impact-first selects all graph-identified affected cases in this demo,
  with a smaller review reduction.
- Hybrid sends low-risk CASE-006 to manual triage instead of silently
  skipping it. Triage is not equivalent to completed re-audit.

## Limitations

- Only six synthetic cases are evaluated.
- Graph-derived coverage does not establish real-world violation recall.
- No claim of legal correctness or production safety is established.
- Regulatory source passages, effective dates, and obligation mappings
  require verification and appropriate human review.
- Results should be rerun on independently annotated cases before making
  comparative performance claims.
