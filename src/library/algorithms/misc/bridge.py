import numpy as np
from numpy.typing import NDArray

from src.library.graph.verification import connected, undirected_only
from src.library.graph.graph import Graph


def dfs(vertex: int, graph: Graph, visit: np.array, visit_time: np.array, low: np.array, parent: np.array):
    visit[vertex] = True
    low[vertex] = visit_time[vertex]
    for neighbor in graph.neighbours(vertex):
        if not visit[neighbor]:
            visit_time[neighbor] = visit_time[vertex] + 1
            parent[neighbor] = vertex
            dfs(neighbor, graph, visit, visit_time, low, parent)
            low[vertex] = min(low[vertex], low[neighbor])
        else:
            if neighbor != parent[vertex]:
                low[vertex] = min(low[vertex], visit_time[neighbor])


@connected
@undirected_only
def bridge(graph: Graph) -> list[tuple[int, int]]:

    n = graph.order
    visit = np.full(n, False, bool)
    visit_time = np.full(n, n, int)
    low = np.full(n, n, int)
    parent = np.full(n, -1, int)

    visit_time[0] = 0
    dfs(0, graph, visit, visit_time, low, parent)

    bridges = []
    for vertex in range(1, n):
        if low[vertex] == visit_time[vertex]:
            bridges.append((vertex, parent[vertex]))
    return bridges

