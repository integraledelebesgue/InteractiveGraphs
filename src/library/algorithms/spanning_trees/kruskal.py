from typing import Optional

import numba

from src.library.graph.verification import connected, positive_weights, undirected_only
from src.library.graph.graph import Graph, Tracker, TrackerCategory


@numba.jit(nopython=False, forceobj=True)
def find(parents: list[int], x: int) -> int:
    if parents[x] != x:
        parents[x] = find(parents, parents[x])
    return parents[x]


@numba.jit(nopython=False, forceobj=True)
def union(parents: list[int], ranks: list[int], x: int, y: int) -> bool:
    x = find(parents, x)
    y = find(parents, y)

    if x == y:
        return False

    if ranks[x] > ranks[y]:
        parents[y] = x

    else:
        parents[x] = y
        if ranks[x] == ranks[y]:
            ranks[y] += 1

    return True


@connected
@positive_weights
@undirected_only
@numba.jit(nopython=False, forceobj=True)
def kruskal(graph: Graph, tracker: Optional[Tracker] = None) -> list[tuple[int, int]]:
    edges = graph.edges
    weights = [graph.adj_matrix[edge] for edge in edges]
    parents = list(range(graph.order))
    ranks = [0 for _ in range(graph.order)]
    spanning_tree_edges = []

    if tracker is not None:
        tracker.add(spanning_tree_edges, TrackerCategory.EDGE_LIST)
        tracker.update()

    for _, (a, b) in sorted(zip(weights, edges)):
        if union(parents, ranks, parents[a], parents[b]):
            spanning_tree_edges.append((a, b))

        if tracker is not None:
            tracker.update()

    return spanning_tree_edges
