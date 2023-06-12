import dataclasses
import itertools
import pickle
import re
import threading
from copy import deepcopy
from enum import Enum
from functools import cached_property, singledispatchmethod
from time import sleep
from typing import Optional, Union, Callable, Any, Iterable, Iterator

import numpy as np

from src.library.algorithms.drawing.spring_embedder import spring_embedder
from src.library.graph.representations import list_to_matrix, matrix_to_list
from src.library.graph.verification import verify_args, ArgumentError


class Graph:
    """An immutable graph"""

    _adj_list: Optional[list[np.ndarray[int]]]
    _adj_matrix: Optional[np.ndarray[int]]
    _weighted: bool
    _directed: bool
    _null_weight: int

    @verify_args
    def __init__(
            self, *,
            adj_list: Optional[list[np.ndarray[int]]] = None,
            adj_matrix: Optional[np.ndarray[int]] = None,
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
        return len(self._adj_list) \
            if self._adj_list is not None \
            else self._adj_matrix.shape[0]

    @cached_property
    def size(self) -> int:
        if self._adj_list is not None:
            return np.concatenate(self._adj_list).size // (1 if self.directed else 2)
        else:
            adj_matrix_flat = self._adj_matrix.flatten()
            return adj_matrix_flat[adj_matrix_flat != self._null_weight].size // (1 if self.directed else 2)

    @cached_property
    def adj_list(self) -> list[np.ndarray[int]]:
        if self._adj_list is None:
            self._adj_list = matrix_to_list(self._adj_matrix, self._null_weight)

        return self._adj_list

    @cached_property
    def adj_matrix(self) -> np.ndarray[int]:
        if self._adj_matrix is None:
            self._adj_matrix = list_to_matrix(self._adj_list, self._null_weight)

        return self._adj_matrix

    @cached_property
    def edges(self) -> np.ndarray[tuple[int, int]]:
        to_search = self._adj_matrix\
             if self._directed\
             else np.triu(self._adj_matrix, 0)

        return np.fromiter(
            zip(*np.where(to_search != self._null_weight)),
            dtype=object
        )

    def neighbours(self, vertex: int) -> np.ndarray[int]:
        if self._adj_list is not None:
            return self._adj_list[vertex]
        if self._adj_matrix is not None:
            return np.where(self._adj_matrix[vertex] != self._null_weight)[0]

    @cached_property
    def connected(self) -> bool:
        return self._quick_dfs()

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
    _adj_list: list[np.ndarray[int]]
    _adj_matrix: np.ndarray[int]
    _weighted: bool
    _directed: bool
    _null_weight: int
    __lazy_vertices: list[list[int]]
    __lazy_edges: tuple[list[int], list[int]]
    __lazy_weights: list[int]
    __modified: bool

    def __init__(self, *args, **kwargs):
        match args:
            case [graph] if isinstance(graph, Graph):
                self.__init_from_graph(graph)
            case []:
                super().__init__(**kwargs)
                self.__init_from_graph(super())
            case _:
                raise ArgumentError(f'No constructor matching arguments {list(map(type, args))}')

        self.__lazy_vertices = [[] for _ in range(2 * len(self._adj_list))]
        self.__lazy_edges = ([], [])
        self.__lazy_weights = []
        self.__modified = False

    def __init_from_graph(self, graph: Graph) -> None:
        self._adj_list = deepcopy(graph.adj_list)
        self._adj_matrix = graph.adj_matrix.copy()
        self._weighted = graph.weighted
        self._directed = graph.directed
        self._null_weight = graph.null_weight

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
    def adj_list(self) -> list[np.ndarray[int]]:
        return self._adj_list

    @property
    @lazy
    def adj_matrix(self) -> np.ndarray[int]:
        return self._adj_matrix

    @property
    @lazy
    def edges(self) -> np.ndarray[tuple[int, int]]:
        to_search = self._adj_matrix\
            if self._directed\
            else np.triu(self._adj_matrix)

        return np.fromiter(
            zip(*np.where(to_search != self._null_weight)),
            dtype=object
        )

    @property
    @lazy
    def connected(self) -> bool:
        return self._quick_dfs()

    def neighbours(self, vertex: int) -> np.ndarray[int]:
        return self._adj_list[vertex]

    def add_vertex(self, neighbours: np.ndarray[int] | list[int]) -> None:
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


class TrackerCategory(Enum):
    QUEUE = 1
    STACK = 2
    TREE = 3
    EDGE_LIST = 4
    VISITED = 5
    DISTANCE = 6
    CURRENT = 7


@dataclasses.dataclass
class Tracker:
    __tracked: list[tuple[TrackerCategory, Any, list[Any]]] = dataclasses.field(default_factory=list)

    def add(self, item: Any, category: TrackerCategory):
        self.__tracked.append((category, item, list()))

    def reset(self):
        self.__tracked.clear()

    def update(self):
        for category, item, history in self.__tracked:
            history.append(
                item.copy()
                if category != TrackerCategory.CURRENT
                else item[0]
            )

    def as_animation_of(self, graph_view: 'GraphView') -> 'Animation':
        return Animation(graph_view, self)

    @property
    def tracked(self) -> list[tuple[TrackerCategory, list[Any]]]:
        return [(category, history) for category, _, history in self.__tracked]


@dataclasses.dataclass
class Frame:
    node_colors: dict[int, str]
    node_labels: dict[int, str]
    edge_colors: dict[tuple[int, int], str]


class ElementCategory(Enum):
    DEFAULT_NODE = 0
    VISITED_NODE = 1
    AWAITING_NODE = 2
    CURRENT_NODE = 3
    DEFAULT_EDGE = 4
    VISITED_EDGE = 5
    AWAITING_EDGE = 6
    CURRENT_EDGE = 7
    MARKED_EDGE = 8


class Animation:
    __graph_view: 'GraphView'
    __tracker: Tracker
    __frames: list[Frame]
    __animation: itertools.cycle
    __current: Frame

    __colormap: dict[ElementCategory, str] = {
        ElementCategory.DEFAULT_NODE: '#bdbdbd',
        ElementCategory.VISITED_NODE: '#8bb162',
        ElementCategory.AWAITING_NODE: '#ad60ba',
        ElementCategory.CURRENT_NODE: '#f3764f',
        ElementCategory.DEFAULT_EDGE: '#000000',
        ElementCategory.MARKED_EDGE: '#8bb162'
    }

    #        ElementCategory.VISITED_EDGE: '#8bb162',
    #        ElementCategory.AWAITING_EDGE: '#ad60ba',
    #        ElementCategory.CURRENT_EDGE: '#f3764f',

    def __init__(self, graph_view: 'GraphView', tracker: Tracker):
        self.__graph_view = graph_view
        self.__frames = self.__build_frames(tracker.tracked)
        self.__animation = itertools.cycle(self.__frames)
        self.__current = self.__animation.__next__()

    @staticmethod
    def __label_steps(
            tracked: list[tuple[TrackerCategory, list[Any]]]
    ) -> Iterator[list[tuple[TrackerCategory, Any]]]:
        return (
            [(category, state) for state in history]
            for category, history in tracked
        )

    def __build_frames(self, tracked: list[tuple[TrackerCategory, list[Any]]]) -> list[Frame]:
        return [
            self.__transform(*step)
            for step in zip(*self.__label_steps(tracked))
        ]

    @staticmethod
    def __tree_to_edges(tree: Iterable[int]) -> list[tuple[int, int]]:
        return [
            (vertex, predecessor)
            for vertex, predecessor in reversed(list(enumerate(tree)))
            if predecessor != -1
        ]

    def __transform(self, *tracking_step: Iterable[tuple[TrackerCategory, Any]]) -> Frame:
        node_colors = dict(
            (index, self.__colormap[ElementCategory.DEFAULT_NODE])
            for index in range(self.__graph_view.graph.order)
        )

        node_labels = dict(
            (index, "")
            for index in range(self.__graph_view.graph.order)
        )

        edge_colors = dict(
            (edge, self.__colormap[ElementCategory.DEFAULT_EDGE])
            for edge in self.__graph_view.graph.edges
        )

        for category, state in tracking_step:
            match category, state:
                case (TrackerCategory.STACK, stack):
                    for vertex in stack:
                        node_colors[vertex] = self.__colormap[ElementCategory.AWAITING_NODE]

                case (TrackerCategory.QUEUE, queue):
                    for vertex in queue:
                        node_colors[vertex] = self.__colormap[ElementCategory.AWAITING_NODE]

                case (TrackerCategory.CURRENT, current):
                    node_colors[current] = self.__colormap[ElementCategory.CURRENT_NODE]

                case (TrackerCategory.DISTANCE, distances):
                    for vertex, distance in enumerate(distances):
                        node_labels[vertex] = str(distance)

                case (TrackerCategory.VISITED, visited):
                    for vertex, is_visited in enumerate(visited):
                        if not is_visited:
                            continue

                        node_colors[vertex] = self.__colormap[ElementCategory.VISITED_NODE]

                case (TrackerCategory.EDGE_LIST, edges):
                    for edge in edges:
                        edge_colors[edge] = self.__colormap[ElementCategory.MARKED_EDGE]

                case (TrackerCategory.TREE, predecessors):
                    edges = self.__tree_to_edges(predecessors)

                    for edge in edges:
                        edge_colors[*edge] = self.__colormap[ElementCategory.MARKED_EDGE]

        return Frame(
            node_colors,
            node_labels,
            edge_colors
        )

    def apply(self):
        self.__graph_view.color_nodes(self.__current.node_colors)
        self.__graph_view.label_nodes(self.__current.node_labels)
        self.__graph_view.color_edges(self.__current.edge_colors)

    def next(self):
        self.__current = self.__animation.__next__()

    def reset(self):
        self.__animation = itertools.cycle(self.__frames)
        self.__current = self.__animation.__next__()

    @staticmethod
    def __validate_color(color: str) -> None:
        if re.match('^#[a-f0-9]{6}$', color) is None:
            raise ArgumentError(f'Color code {color} is invalid')

    @classmethod
    def set_default_colormap(cls):
        pass

    def set_colormap(self):
        pass

    def player(self) -> 'AnimationPlayer':
        return AnimationPlayer(self)

    @property
    def frames(self):
        return self.__frames


class AnimationPlayer(threading.Thread):
    animation: Animation
    __delay: float
    __running: bool
    __paused: bool
    __resume: threading.Condition

    def __init__(self, animation: Animation, delay: float = 0.5):
        self.set_delay(delay)

        super().__init__()

        self.animation = animation
        self.__running = False
        self.__paused = False
        self.__resume = threading.Condition()

    def set_delay(self, delay: float) -> 'AnimationPlayer':
        if delay <= 0:
            raise ArgumentError(f"Expected positive delay value, got {delay}")

        self.__delay = delay
        return self

    def run(self):
        self.__running = True

        while self.__running:
            with self.__resume:
                if self.__paused:
                    self.__resume.wait()

            if not self.__running:
                break

            self.animation.apply()
            sleep(self.__delay)
            self.animation.next()

    def pause(self):
        self.__paused = True

    def resume(self):
        self.__paused = False

        with self.__resume:
            self.__resume.notify()

    def stop(self):
        self.__running = False

        with self.__resume:
            self.__resume.notify()


class Node:
    position: np.ndarray[int]
    vertex: int
    color: str = '#bdbdbd'
    label: str = ""

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
    __edge_colors: np.ndarray[str]
    __default_edge_color: str = '#000000'

    def __init__(self, graph: Graph | MutableGraph, canvas_size: Optional[tuple[int, int]] = None):
        self.__graph = graph \
            if isinstance(graph, MutableGraph) \
            else graph.as_mutable()

        self.__canvas = canvas_size \
            if canvas_size is not None \
            else (1000, 1000)

        self.__nodes = dict(zip(
            range(self.__graph.order),
            map(
                lambda vertex: Node(vertex, tuple(np.random.uniform(0, min(self.__canvas), 2))),
                range(self.__graph.order)
            )
        ))

        self.__edges = list(map(lambda edge: (self.__nodes[edge[0]], self.__nodes[edge[1]]), self.__graph.edges))
        self.__edge_colors = np.full_like(
            self.__graph.adj_matrix,
            self.__default_edge_color,
            dtype=object
        )

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
    def neighbours(self, node: Node | int) -> list[Node]:
        pass

    @neighbours.register(Node)
    def _node_neighbours(self, node: Node) -> list[Node]:
        return list(map(lambda vertex: self.__nodes[vertex], self.__graph.neighbours(node.vertex)))

    @neighbours.register(int)
    def _node_neighbours(self, index: int) -> list[Node]:
        return list(map(lambda vertex: self.__nodes[vertex], self.__graph.neighbours(index)))

    def distribute(self, ideal_length: int) -> None:
        spring_embedder(
            self,
            ideal_length
        )

    def add_node(self, position) -> None:
        self.__graph.add_vertex([])
        number = len(self.__nodes)
        self.__nodes[number] = Node(number, position)

    def add_edge(self, start, end) -> None:
        self.__graph.add_edge(start, end)
        self.__edges.append((start, end))

    def color_nodes(self, colors: dict[int, str]) -> None:
        for index, node in self.__nodes.items():
            node.color = colors[index]

    def label_nodes(self, labels: dict[int, str]) -> None:
        for index, node in self.__nodes.items():
            node.label = str(labels[index])

    def color_edges(self, colors: dict[tuple[int, int], str]):
        for edge in self.graph.edges:
            self.__edge_colors[edge] = colors[edge]

    @property
    def edge_colors(self) -> np.ndarray[str]:
        return self.__edge_colors
