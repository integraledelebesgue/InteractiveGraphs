import ctypes
from typing import Optional

import numpy as np

from src.library.graph.verification import connected, positive_weights, undirected_only
from src.library.graph.graph import Graph, Tracker, TrackerCategory

from heapq import heappop, heappush


def tree_to_edges(parents: np.ndarray[int]) -> list[tuple[int, int]]:
    return [(vertex, parent) for vertex, parent in enumerate(parents) if parent != -1]


@connected
@positive_weights
@undirected_only
def prim(
        graph: Graph,
        start: int = 0,
        tracker: Optional[Tracker] = None
) -> np.ndarray[int]:
    dist = np.full(graph.order, float("inf"), float)
    visit = np.full(graph.order, False, bool)
    spanning_tree_parents = np.full(graph.order, -1, int)

    dist[start] = 0
    pq = [(0, start)]

    vertex = ctypes.c_longlong(start)

    if tracker is not None:
        tracker.add(dist, TrackerCategory.DISTANCE)
        tracker.add(spanning_tree_parents, TrackerCategory.TREE)
        tracker.add(pq, TrackerCategory.QUEUE)
        tracker.add(visit, TrackerCategory.VISITED)
        tracker.add(ctypes.pointer(vertex), TrackerCategory.CURRENT)

    while len(pq) > 0:
        _, vertex.value = heappop(pq)

        if tracker is not None:
            tracker.update()

        if not visit[vertex.value]:
            for neighbor in graph.neighbours(vertex.value):
                weight = graph.adj_matrix[vertex.value, neighbor]
                if not visit[neighbor] and dist[neighbor] > weight:
                    dist[neighbor] = weight
                    spanning_tree_parents[neighbor] = vertex.value
                    heappush(pq, (weight, neighbor))
            visit[vertex.value] = True

    if tracker is not None:
        tracker.update()

    return spanning_tree_parents
