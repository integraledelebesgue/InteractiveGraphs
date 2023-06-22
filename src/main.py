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

view = graph.view()

view.distribute(100)

print([node.position for node in view.nodes.values()])
