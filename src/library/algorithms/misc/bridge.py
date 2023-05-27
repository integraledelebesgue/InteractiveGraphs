from typing import Optional

import numpy as np
from numpy.typing import NDArray

from src.library.graph.verification import connected, undirected_only
from src.library.graph.graph import Graph, Tracker, TrackerCategory


def __dfs(
        vertex: int,
        graph: Graph,
        visit: np.ndarray[bool],
        visit_time: np.ndarray[int],
        low: np.ndarray[int],
        parent: np.ndarray[int]
) -> None:
    visit[vertex] = True
    low[vertex] = visit_time[vertex]
    for neighbor in graph.neighbours(vertex):
        if not visit[neighbor]:
            visit_time[neighbor] = visit_time[vertex] + 1
            parent[neighbor] = vertex
            __dfs(neighbor, graph, visit, visit_time, low, parent)
            low[vertex] = min(low[vertex], low[neighbor])
        else:
            if neighbor != parent[vertex]:
                low[vertex] = min(low[vertex], visit_time[neighbor])


@connected
@undirected_only
def bridge(graph: Graph, start: int = 0, tracker: Optional[Tracker] = None) -> list[tuple[int, int]]:
    n = graph.order
    visit = np.full(n, False, bool)
    visit_time = np.full(n, n, int)
    low = np.full(n, n, int)
    parent = np.full(n, -1, int)
    vertex = None
    bridges = []

    visit_time[0] = 0
    __dfs(start, graph, visit, visit_time, low, parent)

    if tracker is not None:
        tracker.add(visit, TrackerCategory.VISITED)
        tracker.add(parent, TrackerCategory.TREE)
        tracker.add(low, TrackerCategory.DISTANCE)
        tracker.add(vertex, TrackerCategory.CURRENT)

    for vertex in range(n):
        if vertex == start:
            continue

        if tracker is not None:
            tracker.update()

        if low[vertex] == visit_time[vertex]:
            bridges.append((vertex, parent[vertex]))

    if tracker is not None:
        tracker.update()

    return bridges

