import matplotlib.pyplot as plt
import numpy as np

from src.library.graph.graph import Graph

graph = Graph(
    adj_matrix=np.array([
        [-1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, -1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, -1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, -1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, -1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, -1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, -1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, -1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, -1],
    ]),
    weighted=True,
    null_weight=-1
)

graph_view = graph.view()

positions = list(map(lambda node: node.position, graph_view.nodes.values()))

plt.scatter([x[0] for x in positions], [x[1] for x in positions])

graph_view.distribute(200)

positions = list(map(lambda node: node.position, graph_view.nodes.values()))

plt.scatter([x[0] for x in positions], [x[1] for x in positions])
plt.show()
