import ctypes
from typing import Optional

import numpy as np

from src.library.graph.verification import connected, positive_weights
from src.library.graph.graph import Graph, Tracker, TrackerCategory

from heapq import heappop, heappush


@connected
@positive_weights
def dijkstra(
        graph: Graph,
        start: int = 0,
        tracker: Optional[Tracker] = None
) -> tuple[np.ndarray[int], np.ndarray[int]]:
    dist = np.full(graph.order, float("inf"), float)
    visit = np.full(graph.order, False, bool)
    traversal_tree = np.full(graph.order, -1, int)

    dist[start] = 0
    pq = [(0, start)]

    vertex = ctypes.c_longlong(start)

    if tracker is not None:
        tracker.add(pq, TrackerCategory.QUEUE)
        tracker.add(traversal_tree, TrackerCategory.TREE)
        tracker.add(dist, TrackerCategory.DISTANCE)
        tracker.add(visit, TrackerCategory.VISITED)
        tracker.add(ctypes.pointer(vertex), TrackerCategory.CURRENT)

    while len(pq) > 0:
        _, vertex.value = heappop(pq)

        if tracker is not None:
            tracker.update()

        if not visit[vertex.value]:
            for neighbor in graph.neighbours(vertex.value):
                weight = graph.adj_matrix[vertex.value, neighbor] + dist[vertex.value]
                if not visit[neighbor] and dist[neighbor] > weight:
                    dist[neighbor] = weight
                    traversal_tree[neighbor] = vertex.value
                    heappush(pq, (weight, neighbor))
            visit[vertex.value] = True

    if tracker is not None:
        tracker.update()

    return traversal_tree, dist.astype(int)
