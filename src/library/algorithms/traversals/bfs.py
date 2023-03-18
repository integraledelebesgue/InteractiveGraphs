from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from src.library.graph.graph import Graph


def bfs(graph: Graph, start: int) -> Tuple[NDArray, NDArray, NDArray, bool]:
    distance = np.zeros(graph.order, int).fill(-1)
    visited = np.zeros(graph.order, bool)
    visit_order = []
    queue = [start]

    distance[start] = 0.0

    while len(queue) > 0:
        curr = queue.pop()

        if visited[curr]:
            continue

        visited[curr] = True
        visit_order.append(curr)

        neighbours = graph.neighbours(curr)

        if graph.weighted:
            distance[neighbours] = graph.adj_matrix[curr, neighbours] + distance[curr]
        else:
            distance[neighbours] = distance[curr] + 1

        queue.extend(neighbours)

    return distance, visited, np.array(visit_order), np.all(visited)
