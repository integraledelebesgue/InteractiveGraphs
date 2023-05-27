from typing import Optional

import numpy as np
from numpy.typing import NDArray

from copy import deepcopy
from src.library.graph.verification import connected, positive_weights
from src.library.graph.graph import Graph, Tracker

from collections import deque


def __bfs(
        graph: list[list[int]],
        source: int,
        sink: int,
        residual_network: np.ndarray[int],
        parent: np.ndarray[int]
) -> int:
    n = len(graph)
    visit = np.full(n, False, bool)

    queue = deque()
    queue.append(source)
    visit[source] = True

    min_value = float("inf")
    while queue:
        vertex = queue.popleft()
        if vertex == sink:
            return min_value
        for neighbor in graph[vertex]:
            if not visit[neighbor] and residual_network[vertex, neighbor]:
                visit[neighbor] = True
                parent[neighbor] = vertex
                min_value = min(min_value, residual_network[vertex, neighbor])
                queue.append(neighbor)

    return 0


def update_residual_network(
        source: int,
        sink: int,
        parent: np.ndarray[int],
        residual_network: np.ndarray[int],
        change: int
) -> None:
    vertex = sink
    while vertex != source:
        residual_network[vertex, parent[vertex]] += change
        residual_network[parent[vertex], vertex] -= change
        vertex = parent[vertex]


@connected
@positive_weights
def edmonds_karp(
        graph: Graph,
        source: int,
        sink: int,
        tracker: Optional[Tracker] = None
) -> tuple[int, np.ndarray[int]]:
    residual_network = deepcopy(graph.adj_matrix)
    graph_bfs = [
        list(filter(
            lambda neighbor: graph.adj_matrix[vertex, neighbor] != graph.null_weight or \
                             graph.adj_matrix[neighbor, vertex] != graph.null_weight,
            range(graph.order)
        ))
        for vertex in range(graph.order)
    ]

    parent = np.full(graph.order, -1, int)

    flow = 0
    while (flow_increase := __bfs(graph_bfs, source, sink, residual_network, parent)) > 0:
        update_residual_network(source, sink, parent, residual_network, flow_increase)
        flow += flow_increase
        parent = np.full(graph.order, -1, int)

    return flow, residual_network
