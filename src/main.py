import numpy as np

from src.library.algorithms.minimal_paths.floyd_warshall import floyd_warshall
from src.library.algorithms.traversals.bfs import bfs
from src.library.graph.graph import Graph


graph = Graph(
    adj_matrix=np.array([
        [-1, 1, -1, 2],
        [1, -1, 1, 5],
        [-1, 1, -1, 1],
        [2, 5, 1, -1]
    ]),
    weighted=True,
    null_weight=-1
)

print(graph.adj_list)
print(graph.adj_matrix)
print(graph.order)
print(graph.size)
print(graph.connected)
print(floyd_warshall(graph))
