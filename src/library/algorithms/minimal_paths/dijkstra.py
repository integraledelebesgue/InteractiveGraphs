from typing import Optional

import numpy as np

from src.library.graph.verification import connected, positive_weights
from src.library.graph.graph import Graph, Tracker, TrackerCategory

from heapq import heappop, heappush


@connected
@positive_weights
def dijkstra(
        graph: Graph,
        source: int = 0,
        tracker: Optional[Tracker] = None
) -> tuple[np.ndarray[int], np.ndarray[int]]:
    dist = np.full(graph.order, float("inf"), float)
    visit = np.full(graph.order, False, bool)
    traversal_tree = np.full(graph.order, -1, int)

    dist[source] = 0
    pq = [(0, source)]

    vertex = None

    if tracker is not None:
        tracker.add(pq, TrackerCategory.QUEUE)
        tracker.add(traversal_tree, TrackerCategory.TREE)
        tracker.add(dist, TrackerCategory.DISTANCE)
        tracker.add(visit, TrackerCategory.VISITED)
        tracker.add(vertex, TrackerCategory.CURRENT)

    while len(pq) > 0:
        _, vertex = heappop(pq)

        if tracker is not None:
            tracker.update()

        if not visit[vertex]:
            for neighbor in graph.neighbours(vertex):
                weight = graph.adj_matrix[vertex, neighbor] + dist[vertex]
                if not visit[neighbor] and dist[neighbor] > weight:
                    dist[neighbor] = weight
                    traversal_tree[neighbor] = vertex
                    heappush(pq, (weight, neighbor))
            visit[vertex] = True

    if tracker is not None:
        tracker.update()

    return traversal_tree, dist.astype(int)
