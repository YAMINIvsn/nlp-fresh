from dart.dependency_graph import DependencyGraph


def test_reachability_includes_transitive_dependencies():
    graph = DependencyGraph()
    graph.add_dependency("A", "B")
    graph.add_dependency("B", "C")
    assert graph.reachable({"A"}) == {"A", "B", "C"}
