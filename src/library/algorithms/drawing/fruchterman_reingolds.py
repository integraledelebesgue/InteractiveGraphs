import numpy as np
from numpy.typing import NDArray


def repulsive_force(u, v, ideal_length) -> NDArray:
    diff = u.position - v.position
    print(u.position)
    print(v.position)

    if not (u.position[0] <= 0 or u.position[0] > 0):
        exit(1)

    if not (v.position[0] <= 0 or v.position[0] > 0):
        exit(1)

    return diff / np.dot(diff, diff) * ideal_length ** 2


def attractive_force(u, v, ideal_length) -> NDArray:
    diff = v.position - u.position
    return np.sqrt(np.dot(diff, diff)) * diff / ideal_length


def distribute_fruchterman_reingold(
        graph_view,
        ideal_length: int,
        cooling_factor: float,
        max_iter: int
):
    for i in range(max_iter):
        node_forces = []

        for vertex, node in graph_view.nodes.items():
            node_forces.append(
                np.sum(
                    [repulsive_force(node, other_node, ideal_length)
                     for other_node in graph_view.nodes.values() if other_node is not node]
                ) +
                np.sum(
                    [attractive_force(node, neighbour, ideal_length)
                     for neighbour in graph_view.neighbours(node) if neighbour is not node]
                )
            )

        for node, force in zip(graph_view.nodes.values(), node_forces):
            node.position += force * cooling_factor

        cooling_factor **= i
