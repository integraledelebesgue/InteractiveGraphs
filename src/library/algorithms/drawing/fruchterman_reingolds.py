import math

import numpy as np


def distribute(graph, k=0.1, iterations=50):
    # Initialize node positions
    n = graph.order
    positions = np.random.rand(n, 2)

    # Set the area of the graph
    area = math.sqrt(len(graph))

    # Calculate initial distances between nodes
    distances = np.linalg.norm(positions[:, None] - positions, axis=-1)
    np.fill_diagonal(distances, 1e-5)  # Avoid division by zero

    # Perform the main iterations
    for _ in range(iterations):
        # Calculate attractive forces
        attractive_forces = np.zeros((n, n, 2))
        for i in range(n):
            for j in graph.neighbours(i):
                delta = positions[i] - positions[j]
                distance = distances[i, j]
                attractive_forces[i, j] = (delta / distance) * (distance ** 2) / k
                attractive_forces[j, i] = -attractive_forces[i, j]

        # Calculate repulsive forces
        repulsive_forces = np.zeros((n, n, 2))
        for i in range(n):
            for j in range(i + 1, n):
                delta = positions[i] - positions[j]
                distance = distances[i, j]
                repulsive_forces[i, j] = (delta / distance) * (k ** 2) / distance
                repulsive_forces[j, i] = -repulsive_forces[i, j]

        # Update node positions
        delta_pos = np.sum(attractive_forces, axis=1) + np.sum(repulsive_forces, axis=1)
        positions += delta_pos

        # Limit the maximum displacement to the temperature (area) and cool down
        displacement = np.linalg.norm(delta_pos, axis=1)
        max_displacement = np.max(displacement)
        if max_displacement > 0:
            positions *= area / max_displacement

    # Normalize node positions to fit within the drawing area
    positions -= np.min(positions, axis=0)
    positions /= np.max(positions, axis=0)

    return positions
