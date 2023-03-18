from copy import deepcopy

from numpy.typing import NDArray

from src.library.graph.graph import Graph
from src.library.graph.verification import weighted_only, positive_weights


@weighted_only
def floyd_warshall(graph: Graph) -> NDArray:
    n = graph.order
    distance = deepcopy(graph.adj_matrix)
    distance[distance == 0] = float('inf')

    for u in range(n):
        for v in range(n):
            for w in range(n):
                distance[u, v] = min(
                    distance[u, v],
                    distance[u, w] + distance[w, v]
                )

    return distance
