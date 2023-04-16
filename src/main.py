import threading
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

from src.library.algorithms.traversals.bfs import bfs
from src.library.graph.graph import Graph

graph = Graph(
    adj_matrix=np.array([
        [-1, 1, -1, 1, 1],
        [-1, -1, 1, 1, 1],
        [-1, 1, -1, 1, 1],
        [1, 1, -1, -1, 1],
        [-1, -1, -1, 1, -1]
    ]),
    weighted=False,
    directed=True,
    null_weight=-1
)

mut_graph = graph.as_mutable()
hist = deque()

distance, tree = bfs(graph, 0, hist)

print(distance, tree)
print(*hist, sep='\n')
