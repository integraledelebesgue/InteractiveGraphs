import numpy as np
from typing import Callable, Optional
from numpy.typing import NDArray

from src.library.graph.representations import list_to_matrix


def verify_args(graph_constructor: Callable) -> Callable:
    def verifier(
            self, *,
            adj_list: Optional[NDArray] = None,
            adj_matrix: Optional[NDArray] = None,
            weighted: bool = False,
            directed: bool = False
    ) -> object:
        if adj_list is None and adj_matrix is None:
            raise AttributeError("At least one graph representation must be specified")

        if weighted and adj_matrix is None:
            raise AttributeError("Weighted graph requires specifying an adjacency matrix")

        if adj_matrix is not None:
            if not directed and not np.all(np.abs(adj_matrix - adj_matrix.transpose()) == 0):
                raise AttributeError("Undirected graph requires a symmetric adjacency matrix")
            if not np.all(np.diag(adj_matrix) == 0):
                raise AttributeError("Adjacency matrix with non-zero diagonal elements is not allowed")

        if not weighted and adj_list is not None and adj_matrix is not None:
            if not np.array_equal(list_to_matrix(adj_list), np.sign(adj_matrix)):
                raise AttributeError("Both representations of undirected graph must contain identical graphs")

        return graph_constructor(
            self,
            adj_list=adj_list,
            adj_matrix=adj_matrix,
            weighted=weighted,
            directed=directed
        )

    return verifier


def directed_only(graph_func: Callable) -> Callable:
    def verifier(graph, *args, **kwargs):
        if not graph.directed:
            raise AttributeError(f"Function {graph_func} cannot be performed on an undirected graph")

        return graph_func(graph, *args, **kwargs)

    return verifier


def undirected_only(graph_func: Callable) -> Callable:
    def verifier(graph, *args, **kwargs):
        if graph.directed:
            raise AttributeError(f"Function {graph_func} cannot be performed on a directed graph")

        return graph_func(graph, *args, **kwargs)

    return verifier


def weighted_only(graph_func: Callable) -> Callable:
    def verifier(graph, *args, **kwargs):
        if not graph.weighted:
            raise AttributeError(f"Function {graph_func} cannot be performed on a non-weighted graph")

        return graph_func(graph, *args, **kwargs)

    return verifier


def positive_weights(graph_func: Callable) -> Callable:
    def verifier(graph, *args, **kwargs):
        if not np.all(graph.adj_matrix > 0):
            raise AttributeError(f"Function {graph_func} cannot be performed on a graph with non-positive weights")

        return graph_func(graph, *args, **kwargs)

    return verifier
