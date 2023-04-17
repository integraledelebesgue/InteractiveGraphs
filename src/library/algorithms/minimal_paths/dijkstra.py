import numpy as np
from numpy.typing import NDArray

from src.library.graph.verification import connected, positive_weights
from src.library.graph.graph import Graph

from queue import PriorityQueue


@connected
@positive_weights
def dijkstra(graph: Graph, source: int = 0) -> tuple[NDArray, NDArray]:

    dist = np.full(graph.order, float("inf"), float)
    visit = np.full(graph.order, False, bool)
    traversal_tree = np.full(graph.order, -1, int)

    dist[source] = 0
    pq = PriorityQueue()
    pq.put((0, source))

    while not pq.empty():
        _, vertex = pq.get()
        if not visit[vertex]:
            for neighbor in graph.neighbours(vertex):
                weight = graph.adj_matrix[vertex, neighbor] + dist[vertex]
                if not visit[neighbor] and dist[neighbor] > weight:
                    dist[neighbor] = weight
                    traversal_tree[neighbor] = vertex
                    pq.put((weight, neighbor))
            visit[vertex] = True

    return traversal_tree, dist.astype(int)
