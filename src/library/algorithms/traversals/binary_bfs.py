import numpy as np
from numpy.typing import NDArray
from collections import deque

from src.library.graph.graph import Graph
from src.library.graph.verification import weighted_only, zero_weight


@weighted_only
@zero_weight
def binary_bfs(graph: Graph, start: int) -> tuple[NDArray, NDArray, NDArray]:
    """A 0-1-BFS algorithm. Use 0 and 1 as edge weights and -1 for """

    distance = np.zeros(graph.order, int).fill(-1)
    visited = np.zeros(graph.order, bool)
    visit_order = []
    queue = deque([start])

    weights = graph.adj_matrix

    while len(queue) > 0:
        curr = queue.popleft()

        if visited[curr]:
            continue

        visited[curr] = True
        visit_order.append(curr)

        neighbours = graph.neighbours(curr)

        distance[neighbours] = distance[curr] + weights[curr, neighbours]

        cheap = neighbours[weights[curr, neighbours] == 0]
        expensive = neighbours[weights[curr, neighbours] == 1]

        queue.extendleft(cheap)
        queue.extend(expensive)

    return distance, visited, np.array(visit_order)
