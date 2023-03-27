from numpy.typing import NDArray

from src.library.graph.graph import Graph
from src.library.graph.verification import weighted_only, positive_weights

inf = float('inf')


@weighted_only
@positive_weights
def floyd_warshall(graph: Graph) -> NDArray:
    n = graph.order

    distance = graph.adj_matrix\
        .copy()\
        .astype(float)

    distance[distance == graph.null_weight] = inf

    for u in range(n):
        for v in filter(lambda x: x != u, range(n)):
            for w in filter(lambda x: x != u and x != v, range(n)):
                distance[u, v] = min(
                    distance[u, v],
                    distance[u, w] + distance[w, v]
                )

    distance[distance == inf] = -1

    return distance.astype(int)
