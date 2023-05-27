from collections import deque
from typing import Tuple, Optional

import numba
import numpy as np
from numpy.typing import NDArray

from src.library.graph.graph import Graph, Tracker, TrackerCategory


@numba.jit(nopython=False, forceobj=True)
def dfs(
        graph: Graph,
        start: int = 0,
        tracker: Optional[Tracker] = None
) -> Tuple[np.ndarray[int], np.ndarray[int]]:
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
        curr = queue.pop()

        if tracker is not None:
            tracker.update()

        neighbours = graph.neighbours(curr)
        neighbours = neighbours[visited[neighbours] == False]
        visited[neighbours] = True
        distance[neighbours] = distance[curr] + \
           graph.adj_matrix[curr, neighbours] if graph.weighted\
            else 1

        traversal_tree[neighbours] = curr

        queue.extend(neighbours)

    if tracker is not None:
        tracker.update()

    return distance, traversal_tree
