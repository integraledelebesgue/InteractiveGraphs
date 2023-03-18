from functools import cached_property
import numpy as np
from numpy.typing import NDArray
from typing import Optional

from src.library.graph.representations import list_to_matrix, matrix_to_list
from src.library.graph.verification import verify_args


class Graph:
    """An immutable graph"""

    @verify_args
    def __init__(
            self, *,
            adj_list: Optional[NDArray] = None,
            adj_matrix: Optional[NDArray] = None,
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

        self.__adj_list = adj_list
        self.__adj_matrix = adj_matrix
        self.__weighted = weighted
        self.__directed = directed

    @property
    def weighted(self):
        return self.__weighted

    @property
    def directed(self):
        return self.__directed

    @cached_property
    def order(self) -> int:
        return (
            self.__adj_list if self.__adj_list is not None
            else self.__adj_matrix
        ).size

    @cached_property
    def size(self) -> int:
        if self.__adj_list is not None:
            return np.concatenate(self.__adj_list).size // (1 if self.directed else 2)
        else:
            adj_matrix_flat = self.__adj_matrix.flatten()
            return adj_matrix_flat[adj_matrix_flat > 0].size // (1 if self.directed else 2)

    @cached_property
    def adj_list(self) -> NDArray:
        if self.__adj_list is None:
            self.__adj_list = matrix_to_list(self.__adj_matrix)

        return self.__adj_list

    @cached_property
    def adj_matrix(self) -> NDArray:
        if self.__adj_matrix is None:
            self.__adj_matrix = list_to_matrix(self.__adj_list)

        return self.__adj_matrix

    @cached_property
    def connected(self):
        return self.__quick_dfs()

    def neighbours(self, vertex: int) -> NDArray:
        if self.__adj_list is not None:
            return self.__adj_list[vertex]
        if self.__adj_matrix is not None:
            return np.where(self.__adj_matrix[vertex] > 0)[0]

    def __quick_dfs(self) -> bool:
        visited = np.zeros(self.order, bool)
        stack = [0]

        while len(stack) > 0:
            curr = stack.pop()
            visited[curr] = True
            neighbours = self.neighbours(curr)
            stack.extend(neighbours[visited[neighbours] == False])

        return np.all(visited)
