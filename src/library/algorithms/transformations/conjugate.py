import numba
import numpy as np

from src.library.graph.graph import Graph


@numba.jit(nopython=False, forceobj=True)
def conjugate(graph: Graph) -> Graph:
    null_weight = graph.null_weight

    conjugate_matrix = graph.adj_matrix.copy()
    conjugate_matrix[graph.adj_matrix == null_weight] = 1
    conjugate_matrix[graph.adj_matrix != null_weight] = null_weight

    np.fill_diagonal(conjugate_matrix, -1)

    return Graph(
        adj_matrix=conjugate_matrix,
        weighted=False,
        directed=graph.directed,
        null_weight=null_weight
    )

