import timeit

import numpy as np

from src.library.graph.graph import Graph, Tracker
from src.library.algorithms.transformations.conjugate import conjugate
from src.library.algorithms.spanning_trees.kruskal import kruskal
from src.library.algorithms.spanning_trees.prim import prim
from src.library.algorithms.minimal_paths.dijkstra import dijkstra
from src.library.algorithms.flows.edmonds_karp import edmonds_karp
from src.library.algorithms.misc.bridge import bridge
from src.library.algorithms.misc.articulation_point import articulation_point

matrix1 = np.array([
            [-1, 3, -1, -1, -1, 4, -1, -1, -1],
            [3, -1, 5, -1, -1, 2, -1, -1, -1],
            [-1, 5, -1, 6, -1, -1, -1, -1, -1],
            [-1, -1, 6, -1, -1, 4, -1, -1, -1],
            [-1, -1, -1, -1, -1, 17, 5, -1, -1],
            [4, 2, -1, 4, 17, -1, 8, 9, 10],
            [-1, -1, -1, -1, 5, 8, -1, 3, -1],
            [-1, -1, -1, -1, -1, 9, 3, -1, -1],
            [-1, -1, -1, -1, -1, 10, -1, -1, -1],
        ])

graph = Graph(
    adj_matrix=matrix1,
    weighted=True,
    null_weight=-1
)

print(conjugate(graph).adj_matrix)
print(kruskal(graph))
print(prim(graph))
print(dijkstra(graph))
print(edmonds_karp(graph, 0, 4))
print(bridge(graph))
print(articulation_point(graph))
