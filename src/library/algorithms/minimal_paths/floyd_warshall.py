import ctypes
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

    u = ctypes.c_longlong(-1)
    v = ctypes.c_longlong(-1)
    w = ctypes.c_longlong(-1)

    if tracker is not None:
        tracker.add(distance, TrackerCategory.DISTANCE)
        tracker.add(ctypes.pointer(u), TrackerCategory.CURRENT)
        tracker.add(ctypes.pointer(v), TrackerCategory.CURRENT)
        tracker.add(ctypes.pointer(w), TrackerCategory.CURRENT)

    for u.value in range(n):
        for v.value in filter(lambda x: x != u.value, range(n)):
            for w.value in filter(lambda x: x != u.value and x != v.value, range(n)):
                distance[u.value, v.value] = min(
                    distance[u.value, v.value],
                    distance[u.value, w.value] + distance[w.value, v.value]
                )

                if tracker is not None:
                    tracker.update()

    distance[distance == inf] = -1

    return distance.astype(int)
