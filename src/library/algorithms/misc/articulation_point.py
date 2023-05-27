from typing import Optional

import numpy as np

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
        if visit[neighbor]:
            if neighbor != parent[vertex]:
                low[vertex] = min(low[vertex], visit_time[neighbor])
            continue

        visit_time[neighbor] = visit_time[vertex] + 1
        parent[neighbor] = vertex
        __dfs(neighbor, graph, visit, visit_time, low, parent)
        low[vertex] = min(low[vertex], low[neighbor])


@connected
@undirected_only
def articulation_point(graph: Graph, start: int = 0, tracker: Optional[Tracker] = None) -> list[int]:
    n = graph.order
    visit = np.full(n, False, bool)
    low = np.full(n, n, int)
    parent = np.full(n, -1, int)

    visit_time = np.full(n, n, int)
    visit_time[0] = 0

    articulation_points = []

    vertex = None

    __dfs(start, graph, visit, visit_time, low, parent)

    if tracker is not None:
        tracker.add(low, TrackerCategory.DISTANCE)
        tracker.add(visit, TrackerCategory.VISITED)
        tracker.add(parent, TrackerCategory.TREE)
        tracker.add(vertex, TrackerCategory.CURRENT)

    if len([child for child in graph.neighbours(0) if parent[child] == 0]) > 1:
        articulation_points.append(0)

    for vertex in range(n):
        if vertex == start:
            continue

        if tracker is not None:
            tracker.update()

        for child in filter(lambda neighbor: parent[neighbor] == vertex, graph.neighbours(vertex)):
            if low[child] >= visit_time[vertex]:
                articulation_points.append(vertex)
                break

    if tracker is not None:
        tracker.update()

    return articulation_points
