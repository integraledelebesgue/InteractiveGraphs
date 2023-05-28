import ctypes
from typing import Optional

import numpy as np

from src.library.graph.graph import Graph, Tracker, TrackerCategory
from src.library.graph.verification import connected, undirected_only


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
    vertex = ctypes.c_longlong(start)
    bridges = []

    visit_time[0] = 0
    __dfs(start, graph, visit, visit_time, low, parent)

    if tracker is not None:
        tracker.add(visit, TrackerCategory.VISITED)
        tracker.add(parent, TrackerCategory.TREE)
        tracker.add(low, TrackerCategory.DISTANCE)
        tracker.add(ctypes.pointer(vertex), TrackerCategory.CURRENT)

    for vertex.value in range(n):
        if vertex.value == start:
            continue

        if tracker is not None:
            tracker.update()

        if low[vertex.value] == visit_time[vertex.value]:
            bridges.append((vertex.value, parent[vertex.value]))

    if tracker is not None:
        tracker.update()

    return bridges
