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


@jit(nopython=True)
def merge(list1: list[np.ndarray[int]], list2: list[np.ndarray[int]]) -> list[np.ndarray[int]]:
    return list1 + list2


graph = Graph(
    adj_matrix=np.array([
        [-1, 1, -1, 1, 1],
        [-1, -1, 1, 1, 1],
        [-1, 1, -1, 1, 1],
        [1, 1, -1, -1, 1],
        [-1, -1, -1, 1, -1]
    ]),
    weighted=True,
    directed=True,
    null_weight=-1
)

tracker = Tracker()

binary_bfs(graph, tracker=tracker)

print(*tracker.tracked, sep='\n')

"""
mut_graph = graph.as_mutable()

_ = graph.adj_list
distance, tree = bfs(graph, 0)

print(distance, tree)

graph2 = MutableGraph(
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

distance, tree = bfs(graph2, 0)

print(distance, tree)


print(mut_graph.edges)
"""
