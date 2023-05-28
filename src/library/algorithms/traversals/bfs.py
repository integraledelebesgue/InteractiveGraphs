import ctypes
from collections import deque
from typing import Tuple, Optional

import numpy as np
import numba

from src.library.graph.graph import Graph, Tracker, TrackerCategory


@numba.jit(nopython=False, forceobj=True)
def bfs(
        graph: Graph,
        start: int = 0,
        tracker: Optional[Tracker] = None
) -> Tuple[np.ndarray[int], np.ndarray[int]]:
    visited = np.zeros(graph.order, bool)
    visited[start] = True

    distance = np.full(graph.order, -1)
    distance[start] = 0

    traversal_tree = np.full(graph.order, -1)

    queue: deque[int] = deque([start])

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
        distance[neighbours] = distance[curr.value] + \
           graph.adj_matrix[curr.value, neighbours] if graph.weighted\
            else 1

        traversal_tree[neighbours] = curr.value

        queue.extend(neighbours)

    if tracker is not None:
        tracker.update()

    return distance, traversal_tree
