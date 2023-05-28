from typing import Optional

import numba
import numpy as np
from numba.typed.typedlist import List

from src.library.graph.verification import connected, positive_weights, undirected_only
from src.library.graph.graph import Graph, Tracker, TrackerCategory


@numba.jit(nopython=True)
def find(parents: List[int], x: int) -> int:
    if parents[x] != x:
        parents[x] = find(parents, parents[x])
    return parents[x]


@numba.jit(nopython=True)
def union(parents: List[int], ranks: List[int], x: int, y: int) -> bool:
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
def kruskal(graph: Graph, tracker: Optional[Tracker] = None) -> list[tuple[int, int]]:
    edges = graph.edges
    weights = [graph.adj_matrix[edge] for edge in edges]
    parents = List(range(graph.order))
    ranks = List([0 for _ in range(graph.order)])
    spanning_tree_edges = []

    current_edge = []

    if tracker is not None:
        tracker.add(spanning_tree_edges, TrackerCategory.EDGE_LIST)
        tracker.add(current_edge, TrackerCategory.EDGE_LIST)
        tracker.update()

    for _, (a, b) in sorted(zip(weights, edges)):
        if union(parents, ranks, parents[a], parents[b]):
            spanning_tree_edges.append((a, b))

        if tracker is not None:
            current_edge.append((a, b))
            tracker.update()
            current_edge.pop()

    return spanning_tree_edges


@numba.jit(nopython=True)
def __kruskal_untracked(
        adj_matrix: np.ndarray[int],
        edges: List[tuple[int, int]],
        order: int
) -> list[tuple[int, int]]:
    weights = [adj_matrix[edge] for edge in edges]
    parents = List(range(order))
    ranks = List([0 for _ in range(order)])
    spanning_tree_edges = []

    for _, (a, b) in sorted(zip(weights, edges)):
        if union(parents, ranks, parents[a], parents[b]):
            spanning_tree_edges.append((a, b))

    return spanning_tree_edges
