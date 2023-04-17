from collections import deque
from typing import Tuple, Optional

import numpy as np
from numpy.typing import NDArray

from src.library.graph.graph import Graph, Animation


def bfs(
        graph: Graph,
        start: int = 0,
        animation: Optional[Animation] = None
) -> Tuple[NDArray, NDArray]:
    visited = np.zeros(graph.order, bool)
    traversal_tree = np.full(graph.order, -1)
    distance = np.full(graph.order, -1)
    distance[start] = 0

    queue = deque([start])

    while len(queue) > 0:
        curr = queue.popleft()

        if visited[curr]:
            continue
        visited[curr] = True

        if animation is not None:
            animation.add_frame(
                curr,
                queue,
                visited,
                distance
            )

        neighbours = graph.neighbours(curr)
        neighbours = neighbours[visited[neighbours] == False]

        distance[neighbours] = distance[curr] + \
           graph.adj_matrix[curr, neighbours] if graph.weighted\
            else 1

        traversal_tree[neighbours] = curr

        queue.extend(neighbours)

    if animation is not None:
        animation.add_frame(
            None,
            queue,
            visited,
            distance,
            special=True
        )

    return distance, traversal_tree
