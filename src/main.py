from typing import List

import numpy as np
from numba import jit
from numba.core.typing.asnumbatype import as_numba_type
from numba.misc.dummyarray import Array
from numpy.typing import ArrayLike
from numpy.typing import NDArray


from src.library.algorithms.traversals.bfs import bfs
from src.library.algorithms.traversals.binary_bfs import binary_bfs
from src.library.algorithms.traversals.dfs import dfs
from src.library.graph.graph import Graph, MutableGraph, Tracker


graph = Graph(
    adj_matrix=np.array([
        [-1, 1, -1, 1, -1],
        [1, -1, 1, 1, -1],
        [-1, 1, -1, -1, -1],
        [1, 1, -1, -1, 1],
        [-1, -1, -1, 1, -1]
    ]),
    weighted=True,
    directed=False,
    null_weight=-1
)

mut = graph.as_mutable()

print(mut.connected)
print(mut.order)

v = mut.add_vertex([])

print(mut.connected)

mut.add_edge(v, 0, 1)
mut.add_edge(2, 0, 1)

print(mut.connected)
print(mut.order)

print(mut.adj_matrix)
