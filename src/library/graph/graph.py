import itertools
import pickle
import threading
from collections import deque
from dataclasses import dataclass
from functools import cached_property, singledispatchmethod
from time import sleep
from typing import Optional, List, Union, Callable

import numpy as np
from numpy.typing import NDArray

from src.library.algorithms.drawing.fruchterman_reingolds import distribute_fruchterman_reingold
from src.library.graph.representations import list_to_matrix, matrix_to_list
from src.library.graph.verification import verify_args, ArgumentError


class Graph:
    """An immutable graph"""

    _adj_list: Optional[list[NDArray]]
    _adj_matrix: Optional[NDArray]
    _weighted: bool
    _directed: bool
    _null_weight: int

    @verify_args
    def __init__(
            self, *,
            adj_list: Optional[list[NDArray]] = None,
            adj_matrix: Optional[NDArray] = None,
            weighted: bool = False,
            directed: bool = False,
            null_weight: int = 0
    ):
        """
        Create a graph out of adjacency list and/or matrix given.
        List and matrix must contain identical graphs
        or a neighbourhood relation with edge weights for a weighted graph.

            Params:
                adjacency_list (NDArray[NDArray[int]): an adjacency list

                adjacency_matrix (NDArray[int]): an adjacency matrix; represents edge weights for a weighted graph

                weighted (bool): is the graph weighted

                directed (bool): is the graph directed

                null_weight (int): a non-existing edge weight representation
                used in adjacency matrix for a weighted graph. Either 0 or -1.

            Returns: A graph object
        """

        self._adj_list = adj_list
        self._adj_matrix = adj_matrix
        self._weighted = weighted
        self._directed = directed
        self._null_weight = null_weight

    @property
    def weighted(self) -> bool:
        return self._weighted

    @property
    def directed(self) -> bool:
        return self._directed

    @property
    def null_weight(self) -> int:
        return self._null_weight

    @cached_property
    def order(self) -> int:
        return len(self._adj_list)\
            if self._adj_list is not None\
            else self._adj_matrix.shape[0]

    @cached_property
    def size(self) -> int:
        if self._adj_list is not None:
            return np.concatenate(self._adj_list).size // (1 if self.directed else 2)
        else:
            adj_matrix_flat = self._adj_matrix.flatten()
            return adj_matrix_flat[adj_matrix_flat != self._null_weight].size // (1 if self.directed else 2)

    @cached_property
    def adj_list(self) -> list[NDArray]:
        if self._adj_list is None:
            self._adj_list = matrix_to_list(self._adj_matrix, self._null_weight)

        return self._adj_list

    @cached_property
    def adj_matrix(self) -> NDArray:
        if self._adj_matrix is None:
            self._adj_matrix = list_to_matrix(self._adj_list, self._null_weight)

        return self._adj_matrix

    @cached_property
    def edges(self) -> NDArray:
        return np.array(
            zip(
                *np.where(
                    (self._adj_matrix
                     if self._directed
                     else np.triu(self._adj_matrix, 0))
                    != self._null_weight
                )
            )
        )

    @cached_property
    def connected(self) -> bool:
        return self._quick_dfs()

    def neighbours(self, vertex: int) -> NDArray:
        if self._adj_list is not None:
            return self._adj_list[vertex]
        if self._adj_matrix is not None:
            return np.where(self._adj_matrix[vertex] != self._null_weight)[0]

    def _quick_dfs(self) -> bool:
        visited = np.zeros(self.order, bool)
        stack = [0]

        while len(stack) > 0:
            curr = stack.pop()
            visited[curr] = True
            neighbours = self.neighbours(curr)
            stack.extend(neighbours[visited[neighbours] == False])

        return np.all(visited)

    def as_mutable(self) -> 'MutableGraph':
        return MutableGraph(self)

    def view(self, *args) -> 'GraphView':
        return GraphView(self, *args)

    @classmethod
    def from_file(cls, filepath: str) -> Union['Graph', 'MutableGraph']:
        with open(filepath, 'rb') as file:
            obj = pickle.load(file)

        if not isinstance(obj, cls):
            raise Exception(f'Cannot read object from {filepath} - not a graph')

        return obj

    def to_file(self, filepath: str) -> None:
        with open(filepath, 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)


class MutableGraph(Graph):
    _adj_list: list[NDArray]
    _adj_matrix: NDArray
    _weighted: bool
    _directed: bool
    _null_weight: int
    __lazy_vertices: list[list[int]]
    __lazy_edges: tuple[list[int], list[int]]
    __lazy_weights: list[int]
    __modified: bool

    def __init__(self, *args, **kwargs):
        match list(map(lambda arg: type(arg).__name__, args)):
            case ['Graph']:
                self.__init_from_graph(args[0])
                self.__connected = args[0].connected
            case []:
                super().__init__(*args, **kwargs)
            case _:
                raise ArgumentError(f'Cannot construct mutable graph out of {list(map(type, args))}')

        self.__lazy_vertices = [[] for _ in range(2 * len(self._adj_list))]
        self.__lazy_edges = ([], [])
        self.__lazy_weights = []

        self.__modified = False

    def __init_from_graph(self, graph: Graph) -> None:
        _force_computation = graph.adj_matrix, graph.adj_list, graph.order, graph.size, graph.connected

        for name, value in filter(
                lambda item: item[0].startswith("_"),
                map(
                    lambda item: (item[0].removeprefix(f"_{graph.__class__}"), item[1]),
                    graph.__dict__.items()
                )
        ):
            self.__setattr__(name, value)

    @staticmethod
    def lazy(method: Callable) -> Callable:
        def cacher(self, *args, **kwargs):
            name = f'__lazy_{method}'

            if not hasattr(self, name):
                setattr(self, name, None)

            if self.__modified or getattr(self, name) is None:
                self.__lazy_update()
                setattr(self, name, method(self, *args, **kwargs))

            return getattr(self, name)

        return cacher

    @property
    @lazy
    def order(self) -> int:
        return len(self._adj_list)

    @property
    @lazy
    def adj_list(self) -> list[NDArray]:
        return self._adj_list

    @property
    @lazy
    def adj_matrix(self) -> NDArray:
        return self._adj_matrix

    @property
    @lazy
    def edges(self) -> NDArray:
        return np.fromiter(
            zip(
                *np.where(
                    (self._adj_matrix
                     if self._directed
                     else np.triu(self._adj_matrix, 0))
                    != self._null_weight
                )
            ),
            dtype=object
        )

    @property
    @lazy
    def connected(self) -> bool:
        return self._quick_dfs()

    @lazy
    def neighbours(self, vertex: int) -> NDArray:
        return self._adj_list[vertex]

    def add_vertex(self, neighbours: NDArray[int] | List[int]) -> None:
        pass

    def add_edge(self, start: int, end: int) -> None:
        pass

    def delete_vertex(self, vertex) -> None:
        pass

    def delete_edge(self, start, end) -> None:
        pass

    def change_weight(self, start: int, end: int, new_weight: int) -> None:
        pass

    def __lazy_update(self) -> None:
        if not self.__modified:
            return

        self.__modified = False

        new_vertices_count = len(list(filter(None, self.__lazy_vertices)))

        if new_vertices_count > 0:
            self.__update_adj_list()
            self.__update_adj_matrix(new_vertices_count)

        self._adj_matrix[*self.__lazy_edges] = self.__lazy_weights

    def __update_adj_matrix(self, new_vertices_count):
        self._adj_matrix = np.pad(
            self._adj_matrix,
            (0, new_vertices_count),
            'constant', constant_values=self._null_weight
        )

    def __update_adj_list(self):
        for adj_row, lazy_row in zip(self._adj_list, self.__lazy_vertices):
            np.append(adj_row, lazy_row)
            lazy_row.clear()


@dataclass
class Frame:
    queue: list | deque | NDArray
    distance: Optional[NDArray]
    colors: NDArray
    special: bool = False


class Animation(threading.Thread):
    __frames: list[Frame]
    __frames_seq: itertools.cycle
    __frame: Frame
    __delay: float
    __special_delay: float
    __colormap: dict[str, str]
    __graph_view: 'GraphView'
    __running: bool

    def __init__(self, graph_view: 'GraphView'):
        super().__init__()
        self.__frames = []
        self.running = False
        self.__graph_view = graph_view
        self.__delay = 0.5
        self.__special_delay = 2.0
        self.__colormap = {
            'default': '#9b8bb3',
            'current': '#f3764f',
            'awaiting': '#8bb162',
            'visited': '#ad60ba'
        }

    @property
    def frame(self):
        return self.__frame

    def run(self):
        self.reset()
        self.__running = True

        while self.__running:
            self.__graph_view.color_nodes(self.__frame.colors)
            self.__graph_view.label_nodes(self.__frame.distance)
            sleep(
                self.__delay
                if not self.__frame.special
                else self.__special_delay
            )
            self.__frame = self.__frames_seq.__next__()

    def set_delay(self, value: float) -> 'Animation':
        self.__delay = value
        return self

    def set_special_delay(self, value: float) -> 'Animation':
        self.__special_delay = value
        return self

    def set_colors(
            self,
            default: str,
            current: str,
            awaiting: str,
            visited: str
    ) -> 'Animation':
        self.__colormap = {
            'default': default,
            'current': current,
            'awaiting': awaiting,
            'visited': visited
        }
        return self

    def add_frame(
            self,
            curr: Optional[int],
            queue: Optional[list | deque | NDArray],
            visited: Optional[NDArray],
            distance: Optional[NDArray],
            special: bool = False
    ) -> 'Animation':
        colors = self.map_to_colors(curr, queue, visited)
        self.__frames.append(
            Frame(
                queue,
                distance,
                colors,
                special
            )
        )
        return self

    def map_to_colors(
            self,
            curr:
            Optional[int],
            queue: Optional[list | deque | NDArray],
            visited: Optional[NDArray]
    ) -> NDArray:
        colors = np.fromiter(
            map(
                lambda _: self.__colormap['default'],
                self.__graph_view.nodes
            ),
            dtype=object
        )

        if visited is not None:
            colors[visited] = self.__colormap['visited']
        if queue is not None:
            colors[queue] = self.__colormap['awaiting']
        if curr is not None:
            colors[curr] = self.__colormap['current']

        return colors

    def reset(self) -> 'Animation':
        self.__frame = self.__frames[0]
        self.__frames_seq = itertools.cycle(self.__frames)
        return self

    def stop(self):
        self.__running = False


class Node:
    position: NDArray
    vertex: int
    color: str
    label: str

    def __init__(self, vertex, position: tuple[int, int] | None = None):
        self.position = np.array(position)
        self.vertex = vertex

    @property
    def x(self) -> int:
        return self.position[0]

    @property
    def y(self) -> int:
        return self.position[1]

    def shift(self, offset: tuple[int, int]) -> tuple[int, int]:
        return tuple(self.position + offset)


class GraphView:
    __graph: MutableGraph
    __nodes: dict[int, Node]
    __edges: list[tuple[Node, Node]]

    def __init__(self, graph: Graph | MutableGraph, canvas_size: Optional[tuple[int, int]] = None):
        self.__graph = graph\
            if isinstance(graph, MutableGraph)\
            else graph.as_mutable()

        self.__canvas = canvas_size\
            if canvas_size is not None\
            else (1000, 1000)

        self.__nodes = dict(zip(
            range(self.__graph.order),
            map(
                lambda vertex: Node(vertex, tuple(np.random.uniform(0, min(self.__canvas), 2))),
                range(self.__graph.order)
            )
        ))

        self.__edges = list(map(lambda edge: (self.__nodes[edge[0]], self.__nodes[edge[1]]), self.__graph.edges))

    @property
    def graph(self) -> MutableGraph:
        return self.__graph

    @property
    def nodes(self) -> dict[int, Node]:
        return self.__nodes

    @property
    def edges(self) -> list[tuple[Node, Node]]:
        return self.__edges

    @property
    def canvas(self) -> tuple[int, int]:
        return self.__canvas

    @canvas.setter
    def canvas(self, size: tuple[int, int]) -> None:
        if min(size) > 0:
            self.__canvas = size
        else:
            raise Exception('')

    @classmethod
    def from_file(cls, filepath: str) -> 'GraphView':
        with open(filepath, 'rb') as file:
            obj = pickle.load(file)

        if isinstance(obj, cls):
            return obj

        if not isinstance(obj, Graph):
            raise Exception('')

        return cls(obj)

    def to_file(self, filepath: str) -> None:
        self.__graph.to_file(filepath)

        with open(f'{filepath}_view', 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)

    @singledispatchmethod
    def neighbours(self, node: Node | int) -> List[Node | int]:
        pass

    @neighbours.register(Node)
    def _node_neighbours(self, node: Node) -> List[Node]:
        return list(map(lambda vertex: self.__nodes[vertex], self.__graph.neighbours(node.vertex)))

    @neighbours.register(int)
    def _node_neighbours(self, vertex: int) -> List[Node]:
        return list(map(lambda vtx: self.__nodes[vtx], self.__graph.neighbours(vertex)))

    def distribute(self, ideal_length: int) -> None:
        distribute_fruchterman_reingold(self, ideal_length, 0.9, 500, 1000)

    def add_node(self, position) -> None:
        self.__graph.add_vertex([])
        number = len(self.__nodes)
        self.__nodes[number] = Node(number, position)

    def add_edge(self, start, end) -> None:
        self.__graph.add_edge(start, end)
        self.__edges.append((start, end))

    def color_nodes(self, colors: NDArray) -> None:
        for vertex, node in self.__nodes.items():
            node.color = colors[vertex]

    def label_nodes(self, labels: NDArray) -> None:
        for vertex, node in self.__nodes.items():
            node.label = str(labels[vertex])
