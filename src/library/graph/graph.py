from functools import cached_property
import numpy as np
from numpy.typing import NDArray
from typing import Optional

from src.library.algorithms.traversals.dfs import quick_dfs
from src.library.graph.representations import list_to_matrix, matrix_to_list
from src.library.graph.verification import verify


class Graph:
    """An immutable graph"""

    @verify
    def __init__(
            self, *,
            adjacency_list: Optional[NDArray] = None,
            adjacency_matrix: Optional[NDArray] = None,
            weighted: bool = False,
            directed: bool = False
    ):
        """
        Create a graph out of adjacency list and/or matrix given.
        List and matrix must contain identical graphs
        or a neighbourhood relation with edge weights for a weighted graph.

            Params:
                adjacency_list (NDArray[NDArray[int]): an adjacency list

                adjacency_matrix (NDArray[int]): an adjacency matrix; represents edge weights for a weighted graph

                weighted (bool): is the graph weighted

                directed (bool): is the graph directed

            Returns: A graph object
        """

        self.__adj_list = adjacency_list
        self.__adj_matrix = adjacency_matrix
        self.__weighted = weighted
        self.__directed = directed

    @property
    def weighted(self):
        return self.__weighted

    @property
    def directed(self):
        return self.__directed

    @cached_property
    def size(self) -> int:
        return (
            self.__adj_list if self.__adj_list is not None
            else self.__adj_matrix
        ).size

    @cached_property
    def order(self) -> int:
        if self.__adj_list is not None:
            return np.concatenate(self.__adj_list).size // (1 if self.directed else 2)
        else:
            adj_matrix_flat = self.__adj_matrix.flatten()
            return adj_matrix_flat[adj_matrix_flat > 0].size // (1 if self.directed else 2)

    @cached_property
    def adjacency_list(self) -> NDArray:
        if self.__adj_list is None:
            self.__adj_list = matrix_to_list(self.__adj_matrix)

        return self.__adj_list

    @cached_property
    def adjacency_matrix(self) -> NDArray:
        if self.__adj_matrix is None:
            self.__adj_matrix = list_to_matrix(self.__adj_list)

        return self.__adj_matrix

    @cached_property
    def connected(self):
        return quick_dfs(self)

    def neighbours(self, vertex: int) -> NDArray:
        if self.__adj_list is not None:
            return self.__adj_list[vertex]
        if self.__adj_matrix is not None:
            return np.where(self.__adj_matrix[vertex] > 0)[0]
