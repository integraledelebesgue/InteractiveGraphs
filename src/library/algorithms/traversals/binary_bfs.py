import ctypes
from typing import Tuple, Optional

import numba
import numpy as np
from numpy.typing import NDArray
from collections import deque

from src.library.graph.graph import Graph, TrackerCategory, Tracker
from src.library.graph.verification import weighted_only, zero_weight


@weighted_only
@zero_weight
def binary_bfs(
        graph: Graph,
        start: int = 0,
        tracker: Optional[Tracker] = None
) -> Tuple[NDArray, NDArray]:
    visited = np.zeros(graph.order, bool)
    visited[start] = True

    distance = np.full(graph.order, -1)
    distance[start] = 0

    traversal_tree = np.full(graph.order, -1)

    queue = deque([start])

    curr = ctypes.c_longlong(start)

    if tracker is not None:
        tracker.add(queue, TrackerCategory.QUEUE)
        tracker.add(distance, TrackerCategory.DISTANCE)
        tracker.add(traversal_tree, TrackerCategory.TREE)
        tracker.add(ctypes.pointer(curr), TrackerCategory.CURRENT)

    while len(queue) > 0:
        curr.value = queue.popleft()

        if tracker is not None:
            tracker.update()

        neighbours = graph.neighbours(curr.value)
        neighbours = neighbours[visited[neighbours] == False]
        visited[neighbours] = True

        neighbour_distances = graph.adj_matrix[curr.value, :]

        distance[neighbours] = distance[curr.value] + neighbour_distances[neighbours]

        traversal_tree[neighbours] = curr.value

        queue.extendleft(neighbours[neighbour_distances[neighbours] == 0])
        queue.extend(neighbours[neighbour_distances[neighbours] == 1])

    if tracker is not None:
        tracker.update()

    return distance, traversal_tree
