"""Small dependency graph using standard Python, avoiding an extra graph dependency."""
from collections import defaultdict, deque


class DependencyGraph:
    """Directed graph: changed obligation -> obligations/audit rules it can affect."""

    def __init__(self):
        self.edges = defaultdict(set)

    def add_dependency(self, source_id: str, dependent_id: str) -> None:
        if not source_id or not dependent_id:
            raise ValueError("Dependency IDs must be non-empty.")
        self.edges[source_id].add(dependent_id)

    def reachable(self, changed_ids: set[str]) -> set[str]:
        seen = set(changed_ids)
        queue = deque(changed_ids)
        while queue:
            node = queue.popleft()
            for nxt in self.edges.get(node, set()):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return seen


def build_graph(dependencies: list[dict]) -> DependencyGraph:
    graph = DependencyGraph()
    for row in dependencies:
        graph.add_dependency(str(row["source_id"]), str(row["dependent_id"]))
    return graph
