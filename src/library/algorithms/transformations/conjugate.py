import numpy as np

from src.library.graph.graph import Graph


def conjugate(graph: Graph) -> Graph:
    null_weight = graph.null_weight

    conjugate_matrix = np.vstack(
        [list(map(lambda weight: 1 if weight == null_weight else null_weight, row))
         for row in graph.adj_matrix]
    )

    np.fill_diagonal(conjugate_matrix, -1)

    return Graph(
        adj_matrix=conjugate_matrix,
        weighted=False,
        directed=graph.directed,
        null_weight=null_weight
    )

