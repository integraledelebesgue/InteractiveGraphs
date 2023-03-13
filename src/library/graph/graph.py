from functools import cached_property
import numpy as np
from numpy.typing import NDArray
from typing import Optional

from src.library.graph.representations import list_to_matrix, matrix_to_list


class Graph:  # basic non-empty, immutable graph
    def __init__(
            self, *, adjacency_list: Optional[NDArray] = None,
            adjacency_matrix: Optional[NDArray] = None,
            weighted=False,
            directed=False
    ):
        if adjacency_list is None and adjacency_matrix is None:
            raise AttributeError("At least one graph representation must be specified")

        self._adjacency_list: Optional[NDArray] = adjacency_list
        self._adjacency_matrix: Optional[NDArray] = adjacency_matrix

    @cached_property
    def size(self) -> int:
        return (
            self._adjacency_list if self._adjacency_list is not None
            else self._adjacency_matrix
        ).size

    @cached_property
    def order(self) -> int:
        if self._adjacency_list is not None:
            return np.concatenate(self._adjacency_list).size / 2
        else:
            adj_matrix_flat = self._adjacency_matrix.flatten()
            return adj_matrix_flat[adj_matrix_flat > 0].size / 2

    @cached_property
    def adjacency_list(self) -> NDArray:
        if self._adjacency_list is None:
            self._adjacency_list = matrix_to_list(self._adjacency_matrix)

        return self._adjacency_list

    @cached_property
    def adjacency_matrix(self) -> NDArray:
        if self._adjacency_matrix is None:
            self._adjacency_matrix = list_to_matrix(self._adjacency_list)

        return self._adjacency_matrix


graph = Graph()
