import numpy as np
from numpy.typing import NDArray

from src.library.graph.graph import GraphView


def attractive(u, v, ideal_length) -> NDArray:
    diff = v.position - u.position
    return diff * np.linalg.norm(diff) / ideal_length


def repulsive(u, v, ideal_length) -> NDArray:
    diff = u.position - v.position
    return diff / np.dot(diff, diff) * ideal_length ** 2


def force(u, v, ideal_length) -> NDArray:
    return repulsive(u, v, ideal_length) + attractive(u, v, ideal_length)


def distribute(
        graph: GraphView,
        ideal_length: float,
        temperature: float,
        cooling_factor: float,
        max_iterations: int
):
    for i in range(max_iterations):
        for node in graph.nodes.values():
            node.position += temperature * sum([
                force(node, other_node, ideal_length)
                for other_node in graph.nodes.values()
                if other_node is not node
            ])

        temperature *= cooling_factor
