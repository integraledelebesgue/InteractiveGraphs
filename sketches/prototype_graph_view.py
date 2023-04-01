from dataclasses import dataclass


@dataclass
class Node:
    position: tuple

    def shift(self, vec):
        return (
            self.position[0] + vec[0],
            self.position[1] + vec[1]
        )


class PrototypeGraphView:
    def __init__(self):
        self.nodes = [
            Node((200, 100)),
            Node((1000, 700)),
            Node((600, 1000))
        ]

        self.edges = [
            (self.nodes[0], self.nodes[1]),
            (self.nodes[0], self.nodes[2]),
            (self.nodes[1], self.nodes[2])
        ]
