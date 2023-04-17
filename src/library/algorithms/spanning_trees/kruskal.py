from src.library.graph.verification import connected, positive_weights, undirected_only
from src.library.graph.graph import Graph


class Node:
    def __init__(self, node_id):
        self.parent = self
        self.node_id = node_id
        self.rank = 0


def find(x: Node) -> Node:
    if x.parent != x:
        x.parent = find(x.parent)
    return x.parent


def union(x: Node, y: Node) -> bool:
    x = find(x)
    y = find(y)
    if x == y:
        return False
    if x.rank > y.rank:
        y.parent = x

    else:
        x.parent = y
        if x.rank == y.rank:
            y.rank += 1
    return True


@connected
@positive_weights
@undirected_only
def kruskal(graph: Graph) -> list[tuple[int, int]]:

    edges = [(i, j) for i in range(graph.order) for j in graph.neighbours(i) if i < j]
    nodes = [Node(i) for i in range(graph.order)]
    spanning_tree_edges = []

    for a, b in sorted(edges, key=lambda edge: graph.adj_matrix[edge[0], edge[1]]):
        if union(nodes[a], nodes[b]):
            spanning_tree_edges.append((a, b))

    return spanning_tree_edges

