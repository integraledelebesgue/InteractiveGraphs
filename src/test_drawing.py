
import matplotlib.pyplot as plt
import numpy as np

from src.library.graph.graph import Graph
from src.library.graph.graph import GraphView
from src.library.algorithms.drawing.spring_embedder import spring_embedder
from src.library.algorithms.drawing.fruchterman_reingolds_2 import distribute


matrix1 = np.array([
            [-1, 3, -1, -1, -1, 4, -1, -1, -1],
            [3, -1, 5, -1, -1, 2, -1, -1, -1],
            [-1, 5, -1, 6, -1, -1, -1, -1, -1],
            [-1, -1, 6, -1, -1, 4, -1, -1, -1],
            [-1, -1, -1, -1, -1, 17, 5, -1, -1],
            [4, 2, -1, 4, 17, -1, 8, 9, 10],
            [-1, -1, -1, -1, 5, 8, -1, 3, -1],
            [-1, -1, -1, -1, -1, 9, 3, -1, -1],
            [-1, -1, -1, -1, -1, 10, -1, -1, -1],
        ])

matrix2 = np.ones((10, 10))
np.fill_diagonal(matrix2, -1)

graph = Graph(
    adj_matrix=matrix1,
    weighted=True,
    null_weight=-1
)

graph2 = Graph(
    adj_matrix=matrix2,
    weighted=True,
    null_weight=-1
)


def plot_layout(graph_view):
    nodes = graph_view.nodes.values()
    x_values = list(map(lambda node: node.x, nodes))
    y_values = list(map(lambda node: node.y, nodes))
    edges = graph_view.edges
    for a, b in edges:
        plt.plot([a.x, b.x], [a.y, b.y], 'ro-')

    plt.plot(x_values, y_values, 'bo')
    plt.show()

"""
g1_view = GraphView(graph)
plot_layout(g1_view)
spring_embedder(g1_view)
plot_layout(g1_view)

g2_view = GraphView(graph2)
plot_layout(g2_view)
spring_embedder(g2_view)
plot_layout(g2_view)
"""

g2_view = GraphView(graph2)
plot_layout(g2_view)
distribute(g2_view, 100, 0.2, 0.9, 10)
plot_layout(g2_view)
