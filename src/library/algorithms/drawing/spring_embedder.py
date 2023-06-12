import numpy as np


def repulsive(u, v, c_rep):
    vector = v - u
    return (c_rep / np.dot(vector, vector)) * (vector / np.linalg.norm(vector))


def attractive(u, v, c_spring, c_rep, length):
    vector = v - u
    return c_spring*np.log(np.linalg.norm(vector) / length) * (vector / np.linalg.norm(vector)) \
        - repulsive(u, v, c_rep)


def rescale_positions(graph, positions):
    target_x, target_y = graph.canvas
    min_x = np.min(positions[:, 0])
    min_y = np.min(positions[:, 1])
    positions -= np.array([min_x, min_y])
    scale_x = target_x / np.max(positions[:, 0])
    scale_y = target_y / np.max(positions[:, 1])
    positions[:, 0] *= scale_x
    positions[:, 1] *= scale_y


def spring_embedder(graph, length: float = 100,
                    cooling: float = 0.99, c_rep: float = 20, c_spring: float = 20, max_iter: int = 100):

    node_ind = {node: index for index, node in graph.nodes.items()}
    n = len(node_ind)
    g = [list(map(lambda node: node_ind[node], graph.neighbours(i))) for i in range(n)]
    positions = np.array([[graph.nodes[i].x, graph.nodes[i].y] for i in range(n)])

    temperature = 10
    for _ in range(max_iter):
        forces = np.zeros((n, 2))
        for i in range(n):
            for v in g[i]:
                forces[i] += attractive(positions[i], positions[v], c_spring, c_rep, length)
            for v in range(n):
                if v != i:
                    forces[i] += repulsive(positions[i], positions[v], c_rep)
        positions += forces*temperature
        temperature *= cooling

    rescale_positions(graph, positions)
    for i in range(n):
        graph.nodes[i].position = positions[i]
