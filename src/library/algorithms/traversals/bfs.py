from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from src.library.graph.graph import Graph


def bfs(graph: Graph, start: int = 0) -> Tuple[NDArray, NDArray]:
    distance = np.zeros(graph.order, int)
    distance.fill(-1)

    visited = np.zeros(graph.order, bool)

    traversal_tree = np.zeros(graph.order, int)
    traversal_tree.fill(-1)

    queue = [start]

    distance[start] = 0.0
    traversal_tree[start] = start

    while len(queue) > 0:
        curr = queue.pop()

        if visited[curr]:
            continue

        visited[curr] = True

        neighbours = graph.neighbours(curr)
        neighbours = neighbours[visited[neighbours] == False]

        if graph.weighted:
            distance[neighbours] = graph.adj_matrix[curr, neighbours] + distance[curr]
        else:
            distance[neighbours] = distance[curr] + 1

        traversal_tree[neighbours] = curr

        queue.extend(neighbours)

    return distance, traversal_tree
