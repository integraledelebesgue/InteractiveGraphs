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
    distance = np.zeros(graph.order, int)
    distance.fill(-1)

    visited = np.zeros(graph.order, bool)

    traversal_tree = np.zeros(graph.order, int)
    traversal_tree.fill(-1)

    distance[start] = 0.0
    traversal_tree[start] = start

    queue = deque([start])

    while len(queue) > 0:
        curr = queue.popleft()

        if visited[curr]:
            continue

        if animation is not None:
            animation.add_frame(
                curr,
                queue,
                visited,
                distance
            )

        visited[curr] = True

        neighbours = graph.neighbours(curr)
        neighbours = neighbours[visited[neighbours] == False]

        if graph.weighted:
            distance[neighbours] = graph.adj_matrix[curr, neighbours] + distance[curr]
        else:
            distance[neighbours] = distance[curr] + 1

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
