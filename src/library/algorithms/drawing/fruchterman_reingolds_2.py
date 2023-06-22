import numpy as np


def attractive(u, v, ideal_length) -> np.ndarray:
    print(u.position, v.position)
    diff = v.position - u.position
    return diff * np.linalg.norm(diff) / ideal_length


def repulsive(u, v, ideal_length) -> np.ndarray:
    print(u.position, v.position)
    diff = u.position - v.position
    return diff / np.dot(diff, diff) * ideal_length ** 2


def distribute(
        graph: "GraphView",
        ideal_length: float,
        cooling_factor: float,
        n_iterations: int
) -> None:
    temperature = 1.0

    for node in graph.nodes.values():
        node.position = np.random.rand(2) * ideal_length

    for i in range(n_iterations):
        for node in graph.nodes.values():
            node.position += temperature * (
                np.sum([
                    attractive(node, neighbour, ideal_length) for neighbour in graph.neighbours(node.vertex)
                ]) +
                np.sum([
                    repulsive(node, neighbour, ideal_length) for neighbour in graph.nodes.values() if neighbour != node
                ])
            )

        temperature *= cooling_factor
