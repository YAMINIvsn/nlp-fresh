
"""Propagate regulatory changes to compliance cases.

This module:
1. Finds directly and indirectly affected obligations.
2. Records dependency paths for explainability.
3. Identifies potentially affected compliance cases.
4. Runs the existing selective re-audit selector.
"""

from collections import deque
from typing import Any

from .dependency_graph import build_graph
from .selective_reaudit import select_cases


def _find_paths(
    graph: Any,
    changed_ids: set[str],
) -> dict[str, list[str]]:
    """Return one shortest dependency path to each reachable obligation."""

    paths: dict[str, list[str]] = {}
    queue = deque()

    for changed_id in sorted(changed_ids):
        paths[changed_id] = [changed_id]
        queue.append(changed_id)

    while queue:
        current = queue.popleft()

        for dependent in sorted(graph.edges.get(current, set())):
            if dependent not in paths:
                paths[dependent] = paths[current] + [dependent]
                queue.append(dependent)

    return paths


def _parse_obligation_ids(value: Any) -> set[str]:
    """Parse semicolon-separated obligation IDs from a case record."""

    if value is None:
        return set()

    if isinstance(value, (list, tuple, set)):
        return {
            str(item).strip()
            for item in value
            if str(item).strip()
        }

    return {
        item.strip()
        for item in str(value).split(";")
        if item.strip()
    }


def propagate_impact(
    changed_obligation_ids: set[str],
    dependencies: list[dict],
    cases: list[dict],
    risk_threshold: float = 0.35,
) -> dict:
    """Trace regulatory changes to affected cases and re-audit decisions.

    Dependency rows must contain:
        source_id, dependent_id

    Case rows must contain:
        case_id, obligation_ids

    Optional case fields supported by the existing selector:
        risk_score, mandatory_review, uncertain

    A dependency path is interpreted as:
        changed obligation -> dependent obligation/control
    """

    changed_ids = {
        str(item).strip()
        for item in changed_obligation_ids
        if str(item).strip()
    }

    if not changed_ids:
        raise ValueError(
            "At least one changed obligation ID is required."
        )

    graph = build_graph(dependencies)

    # Include changed obligations and all transitive dependents.
    paths = _find_paths(graph, changed_ids)
    affected_ids = set(paths)

    case_impacts = []

    for case in cases:
        case_id = str(case.get("case_id", "")).strip()
        case_obligations = _parse_obligation_ids(
            case.get("obligation_ids", "")
        )

        matched_ids = sorted(case_obligations & affected_ids)

        if not matched_ids:
            continue

        case_impacts.append({
            "case_id": case_id,
            "matched_obligation_ids": matched_ids,
            "impact_paths": [
                {
                    "obligation_id": obligation_id,
                    "path": paths[obligation_id],
                }
                for obligation_id in matched_ids
            ],
            "impact_status": "potentially_affected",
        })

    # Reuse the existing, tested selection logic.
    reaudit = select_cases(
        cases=cases,
        affected_ids=affected_ids,
        risk_threshold=risk_threshold,
    )

    return {
        "changed_obligation_ids": sorted(changed_ids),
        "affected_obligation_ids": sorted(affected_ids),
        "case_impacts": case_impacts,
        "reaudit": reaudit,
        "summary": {
            "changed_obligation_count": len(changed_ids),
            "affected_obligation_count": len(affected_ids),
            "potentially_affected_case_count": len(case_impacts),
            "selected_for_reaudit": reaudit["counts"][
                "selected_cases"
            ],
            "not_selected_for_reaudit": reaudit["counts"][
                "skipped_cases"
            ],
        },
        "warning": (
            "Graph reachability indicates potential impact, not "
            "a definitive legal conclusion. Validate dependencies, "
            "case scope, and regulatory applicability."
        ),
    }
