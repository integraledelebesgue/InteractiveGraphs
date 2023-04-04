from functools import wraps

import numpy as np
from typing import Callable, Optional
from numpy.typing import NDArray

from src.library.graph.representations import list_to_matrix


class ArgumentError(ValueError):
    def __init__(self, message):
        super().__init__(message)


def verify_args(graph_constructor: Callable) -> Callable:
    @wraps(graph_constructor)
    def verifier(
            self, *,
            adj_list: Optional[NDArray] = None,
            adj_matrix: Optional[NDArray] = None,
            weighted: bool = False,
            directed: bool = False,
            null_weight: int = 0
    ) -> object:
        if adj_list is None and adj_matrix is None:
            raise ArgumentError("At least one graph representation must be specified")

        if weighted and adj_matrix is None:
            raise ArgumentError("Weighted graph requires specifying an adjacency matrix")

        if adj_matrix is not None:
            if not directed and not np.all(np.abs(adj_matrix - adj_matrix.transpose()) == 0):
                raise ArgumentError("Undirected graph requires a symmetric adjacency matrix")
            if not np.all(np.diag(adj_matrix) == null_weight):
                raise ArgumentError("Adjacency matrix with self-connected diagonal elements is not allowed")

        if not weighted and adj_list is not None and adj_matrix is not None:
            if not np.array_equal(list_to_matrix(adj_list, null_weight), np.sign(adj_matrix)):
                raise ArgumentError("Both representations of undirected graph must contain identical graphs")

        if null_weight not in {0, -1}:
            raise ArgumentError("A null-weight must be set to either 0 or -1")

        return graph_constructor(
            self,
            adj_list=adj_list,
            adj_matrix=adj_matrix,
            weighted=weighted,
            directed=directed,
            null_weight=null_weight
        )

    return verifier


def directed_only(graph_func: Callable) -> Callable:
    @wraps(graph_func)
    def verifier(graph, *args, **kwargs):
        if not graph.directed:
            raise ArgumentError(f"Function {graph_func} cannot be performed on an undirected graph")

        return graph_func(graph, *args, **kwargs)

    return verifier


def undirected_only(graph_func: Callable) -> Callable:
    @wraps(graph_func)
    def verifier(graph, *args, **kwargs):
        if graph.directed:
            raise ArgumentError(f"Function {graph_func} cannot be performed on a directed graph")

        return graph_func(graph, *args, **kwargs)

    return verifier


def weighted_only(graph_func: Callable) -> Callable:
    @wraps(graph_func)
    def verifier(graph, *args, **kwargs):
        if not graph.weighted:
            raise ArgumentError(f"Function {graph_func} cannot be performed on a non-weighted graph")

        return graph_func(graph, *args, **kwargs)

    return verifier


def positive_weights(graph_func: Callable) -> Callable:
    @wraps(graph_func)
    def verifier(graph, *args, **kwargs):
        if not np.all(graph.adj_matrix >= graph.null_weight):
            raise ArgumentError(f"Function {graph_func} cannot be performed on a graph with non-positive weights")

        return graph_func(graph, *args, **kwargs)

    return verifier


def zero_weight(graph_func: Callable) -> Callable:
    @wraps(graph_func)
    def verifier(graph, *args, **kwargs):
        if graph.null_weight == 0:
            raise ArgumentError(f"Function {graph_func} cannot be performed ona a graph with null-weight equal to 0")

        return graph_func(graph, *args, **kwargs)

    return verifier
