import numpy as np

from src.library.graph.graph import Graph


graph = Graph(
    adjacency_list=np.array([
        np.array([1, 2, 3]),
        np.array([0, 2]),
        np.array([0, 1, 3]),
        np.array([0, 2])
    ], dtype=object),
    adjacency_matrix=np.array([
        [0, 1, 1, 1],
        [1, 0, 1, 0],
        [1, 1, 0, 1],
        [1, 0, 1, 0]
    ])
)

print(graph.adjacency_list)
print(graph.adjacency_matrix)
print(graph.order)
print(graph.size)
