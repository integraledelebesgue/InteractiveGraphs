import numpy as np
from numpy.typing import NDArray


def matrix_to_list(adjacency_matrix: NDArray, null_weight: int) -> list[NDArray]:
    return [np.where(row != null_weight)[0] for row in adjacency_matrix]


def list_to_matrix(adjacency_list: list[NDArray], null_weight: int) -> NDArray:
    matrix = np.zeros((len(adjacency_list), len(adjacency_list)), int)
    matrix.fill(null_weight)

    for vertex, neighbours in enumerate(adjacency_list):
        matrix[vertex, neighbours] = 1

    return matrix
