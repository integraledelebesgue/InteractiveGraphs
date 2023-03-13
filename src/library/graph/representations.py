import numpy as np
from numpy.typing import NDArray


def matrix_to_list(adjacency_matrix: NDArray) -> NDArray:
    return np.array(
        [np.where(row > 0)[0] for row in adjacency_matrix]
    )


def list_to_matrix(adjacency_list: NDArray) -> NDArray:
    matrix = np.zeros((adjacency_list.size, adjacency_list.size), int)

    for vertex, neighbours in enumerate(adjacency_list):
        matrix[vertex, neighbours] = 1

    return matrix
