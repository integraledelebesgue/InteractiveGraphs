from typing import Optional

from numpy.typing import NDArray

from src.library.graph.graph import Graph, Tracker, TrackerCategory
from src.library.graph.verification import weighted_only, positive_weights

inf = float('inf')


@weighted_only
@positive_weights
def floyd_warshall(graph: Graph, tracker: Optional[Tracker] = None) -> NDArray:
    n = graph.order

    distance = graph.adj_matrix\
        .copy()\
        .astype(float)

    distance[distance == graph.null_weight] = inf

    u = None
    v = None
    w = None

    if tracker is not None:
        tracker.add(distance, TrackerCategory.DISTANCE)
        tracker.add(u, TrackerCategory.CURRENT)
        tracker.add(v, TrackerCategory.CURRENT)
        tracker.add(w, TrackerCategory.CURRENT)

    for u in range(n):
        for v in filter(lambda x: x != u, range(n)):
            for w in filter(lambda x: x != u and x != v, range(n)):
                distance[u, v] = min(
                    distance[u, v],
                    distance[u, w] + distance[w, v]
                )

                if tracker is not None:
                    tracker.update()

    distance[distance == inf] = -1

    return distance.astype(int)
