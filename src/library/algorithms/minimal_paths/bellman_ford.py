import numpy as np
from numpy.typing import NDArray

from src.library.graph.verification import connected, positive_weights, directed_only
from src.library.graph.graph import Graph


@connected
@directed_only
@positive_weights
def bellman_ford(graph: Graph, source: int = 0) -> tuple[NDArray, NDArray]:

    dist = np.full(graph.order, float("inf"), float)

    dist[source] = 0
    parents = np.full(graph.order, -1, int)
    edges = [(i, j) for i in range(graph.order) for j in graph.neighbours(i)]

    for _ in range(graph.order - 1):
        for a, b in edges:
            if dist[b] > dist[a] + graph.adj_matrix[a, b]:
                dist[b] = dist[a] + graph.adj_matrix[a, b]
                parents[b] = a

    return parents, dist.astype(int)
