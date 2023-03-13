import numpy as np
from numpy.typing import ArrayLike


def traverse_bfs(graph: ArrayLike, start: int):
    queue = [start]
    visited = np.zeros_like(graph, bool)

    while len(queue) > 0:
        curr = queue.pop()

        if visited[curr]:
            continue

        visited[curr] = True
        print(curr)

        queue.extend(graph[curr])


simple_graph = np.array(
    [
        np.array([2, 3]),
        np.array([0, 2, 3]),
        np.array([0, 1]),
        np.array([1])
    ],
    dtype=object
)

traverse_bfs(simple_graph, 0)
