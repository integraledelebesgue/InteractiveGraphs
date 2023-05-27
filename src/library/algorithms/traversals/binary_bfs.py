from typing import Tuple, Optional

import numba
import numpy as np
from numpy.typing import NDArray
from collections import deque

from src.library.graph.graph import Graph, TrackerCategory, Tracker
from src.library.graph.verification import weighted_only, zero_weight


@numba.jit
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

    curr = None

    if tracker is not None:
        tracker.add(queue, TrackerCategory.QUEUE)
        tracker.add(distance, TrackerCategory.DISTANCE)
        tracker.add(traversal_tree, TrackerCategory.TREE)
        tracker.add(curr, TrackerCategory.CURRENT)

    while len(queue) > 0:
        curr = queue.popleft()

        if tracker is not None:
            tracker.update()

        neighbours = graph.neighbours(curr)
        neighbours = neighbours[visited[neighbours] == False]
        visited[neighbours] = True

        neighbour_distances = graph.adj_matrix[curr, neighbours]

        distance[neighbours] = distance[curr] + neighbour_distances

        traversal_tree[neighbours] = curr

        queue.extendleft(neighbours[neighbour_distances[neighbours] == 0])
        queue.extend(neighbours[neighbour_distances[neighbours] == 1])

    if tracker is not None:
        tracker.update()

    return distance, traversal_tree
