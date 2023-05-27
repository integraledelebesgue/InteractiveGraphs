from typing import Optional

import numpy as np

from src.library.graph.verification import connected, positive_weights, directed_only
from src.library.graph.graph import Graph, Tracker, TrackerCategory


@connected
@directed_only
@positive_weights
def bellman_ford(
        graph: Graph,
        source: int = 0,
        tracker: Optional[Tracker] = None
) -> tuple[np.ndarray[int], np.ndarray[int]]:
    dist = np.full(graph.order, float("inf"), float)
    dist[source] = 0

    parents = np.full(graph.order, -1, int)

    edges = [(i, j) for i in range(graph.order) for j in graph.neighbours(i)]

    curr = None

    if tracker is not None:
        tracker.add(curr, TrackerCategory.CURRENT)
        tracker.add(dist, TrackerCategory.DISTANCE)
        tracker.add(parents, TrackerCategory.TREE)

    for curr in range(graph.order - 1):
        if tracker is not None:
            tracker.update()

        for a, b in edges:
            if dist[b] > dist[a] + graph.adj_matrix[a, b]:
                dist[b] = dist[a] + graph.adj_matrix[a, b]
                parents[b] = a

    if tracker is not None:
        tracker.update()

    return parents, dist.astype(int)
