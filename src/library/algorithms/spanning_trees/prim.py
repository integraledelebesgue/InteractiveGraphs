import numpy as np
from numpy.typing import NDArray

from src.library.graph.verification import connected, positive_weights, undirected_only
from src.library.graph.graph import Graph

from queue import PriorityQueue


@connected
@positive_weights
@undirected_only
def prim(graph: Graph, start: int = 0) -> NDArray:

    dist = np.full(graph.order, float("inf"), float)
    visit = np.full(graph.order, False, bool)
    spanning_tree_parents = np.full(graph.order, -1, int)

    dist[start] = 0
    pq = PriorityQueue()
    pq.put((0, start))

    while not pq.empty():
        _, vertex = pq.get()
        if not visit[vertex]:
            for neighbor in graph.neighbours(vertex):
                weight = graph.adj_matrix[vertex, neighbor]
                if not visit[neighbor] and dist[neighbor] > weight:
                    dist[neighbor] = weight
                    spanning_tree_parents[neighbor] = vertex
                    pq.put((weight, neighbor))
            visit[vertex] = True

    return spanning_tree_parents
