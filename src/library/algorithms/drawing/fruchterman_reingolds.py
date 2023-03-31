import numpy as np
from numpy.typing import NDArray


def repulsive_force(u, v, ideal_length) -> NDArray:
    diff = u.position - v.position
    return diff / np.dot(diff, diff) * ideal_length ** 2


def attractive_force(u, v, ideal_length) -> float:
    diff = v.position - u.position
    return np.sqrt(np.dot(diff, diff)) * diff / ideal_length


def distribute_fruchterman_reingold(
        graph_view,
        ideal_length: int,
        cooling_factor: float,
        max_iter: int,
        eps: float
):
    max_force = float('inf')

    for i in range(max_iter):
        if max_force <= eps:
            break

        max_force = float('-inf')
        node_forces = []

        for vertex, node in graph_view.nodes.items():
            node_forces.append(
                np.sum(
                    [repulsive_force(node, other_node, ideal_length)
                     for other_node in graph_view.nodes.values() if other_node is not node]
                ) + \
                np.sum(
                    [attractive_force(node, neighbour, ideal_length)
                     for neighbour in graph_view.neighbours(node) if neighbour is not node]
                )
            )

        for node, force in zip(graph_view.nodes.values(), node_forces):
            node.position += force * cooling_factor

        max_force = max(max_force, max(map(lambda vec: np.sqrt(np.dot(vec, vec)), node_forces)))

        cooling_factor *= i
