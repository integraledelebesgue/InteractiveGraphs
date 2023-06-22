import threading
from dataclasses import dataclass
from os import getcwd
from typing import Any, Optional

import numpy as np
import pygame
import pygame_gui

from src.library.algorithms.minimal_paths.bellman_ford import bellman_ford
from src.library.algorithms.minimal_paths.dijkstra import dijkstra
from src.library.algorithms.minimal_paths.floyd_warshall import floyd_warshall
from src.library.algorithms.misc.articulation_point import articulation_point
from src.library.algorithms.misc.bridge import bridge
from src.library.algorithms.spanning_trees.kruskal import kruskal
from src.library.algorithms.spanning_trees.prim import prim
from src.library.algorithms.traversals.bfs import bfs
from src.library.algorithms.traversals.binary_bfs import binary_bfs
from src.library.algorithms.traversals.dfs import dfs
from src.library.graph.graph import Graph, GraphView, Tracker, AnimationPlayer, Node
from src.library.graph.verification import ArgumentError

window_size = (1400, 1000)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

test_graph = Graph(
    adj_matrix=np.array([
        [-1, 3, -1, -1, -1, 4, -1, -1, -1],
        [3, -1, 5, -1, -1, 2, -1, -1, -1],
        [-1, 5, -1, 6, -1, -1, -1, -1, -1],
        [-1, -1, 6, -1, -1, 4, -1, -1, -1],
        [-1, -1, -1, -1, -1, 17, 5, -1, -1],
        [4, 2, -1, 4, 17, -1, 8, 9, 10],
        [-1, -1, -1, -1, 5, 8, -1, 3, -1],
        [-1, -1, -1, -1, -1, 9, 3, -1, -1],
        [-1, -1, -1, -1, -1, 10, -1, -1, -1],
    ]),
    weighted=True,
    null_weight=-1
)


def set_handler(self, handler):
    self.handler = handler
    return self


pygame_gui.elements.UIButton.set_handler = set_handler


@dataclass
class Position:
    x: float
    y: float


@dataclass
class ContextMenu:
    active: bool
    container: pygame_gui.core.UIContainer
    row_height: int
    items: list[pygame_gui.elements.UIButton]


@dataclass
class GraphIO:
    input_box: pygame_gui.elements.UITextEntryLine
    active: bool


class App(threading.Thread):
    WINDOW_SIZE: tuple[int, int] = (1400, 1000)
    MAX_QUEUE_LENGTH: int = 10
    QUEUE_ELEMENT_SIZE: int = 30
    QUEUE_ELEMENT_SPACING: int = 10

    NODE_RADIUS: float = 20.0
    NODE_BORDER_WIDTH: int = 3

    def __init__(self):
        super().__init__()
        pygame.init()
        pygame.display.set_caption('Interactive graphs')

        self.window_surface = pygame.display \
            .set_mode(
                self.WINDOW_SIZE,
                pygame.RESIZABLE
            )

        self.background = pygame.Surface(self.WINDOW_SIZE)
        self.background.fill(pygame.Color('#ffffff'))

        self.manager = pygame_gui.UIManager(self.WINDOW_SIZE)
        self.clock = pygame.time.Clock()
        self.running = False

        self.left_margin = 15
        self.right_margin = 15
        self.top_margin = 15
        self.bottom_margin = 15

        self.graph_area_corner = (self.left_margin + 120, self.top_margin)
        self.graph_area_size = (1345, 970)

        self.graph_area = pygame.Rect(
            self.graph_area_corner,
            self.graph_area_size
        )

        self.graph_view = test_graph.view(self.graph_area_size)

        for node in self.graph_view.nodes.values():
            node.shift(self.graph_area_corner)

        # self.graph_view.distribute(50)

        self.player: Optional[AnimationPlayer] = None
        self.tracker: Optional[Tracker] = Tracker()

        self.draggable = []
        self.dragging_allowed = True
        self.mouse_attached = None
        self.mouse_element_offset = None

        self.mouse_x = 0
        self.mouse_y = 0

        self.load_io: Optional[GraphIO] = None
        self.ui_load_graph()

        self.save_io: Optional[GraphIO] = None
        self.ui_save_graph()

        self.choose_algorithm: Optional[ContextMenu] = None
        self.ui_choose_algorithm()

        self.edit_menu: Optional[ContextMenu] = None
        self.ui_edit_graph()

        self.context_menu: Optional[ContextMenu] = None
        self.ui_context_menu()

    def play(self) -> None:
        if self.player is not None:
            self.player.stop()

        self.player = self.tracker \
            .as_animation_of(self.graph_view) \
            .player() \
            .set_delay(1.0)

        self.player.start()

    def stop(self) -> None:
        if self.player is not None:
            self.player.stop()
            self.player = None

    def reset_tracker(self) -> None:
        self.tracker.reset()

    def ui_context_menu(self) -> None:
        context_menu_button_size = (200, 25)
        self.context_menu = ContextMenu(
            container=(cont := pygame_gui.core.UIContainer(
                pygame.Rect(
                    (-200, -200),
                    (200, 125),
                ),
                manager=self.manager
            )),
            row_height=50,
            items=[
                pygame_gui.elements.UIButton(
                    relative_rect=pygame.Rect((0, 0), context_menu_button_size),
                    text='Add node',
                    manager=self.manager,
                    container=cont
                ),
                pygame_gui.elements.UIButton(
                    relative_rect=pygame.Rect((0, 25), context_menu_button_size),
                    text='Delete node',
                    manager=self.manager,
                    container=cont
                ),
                pygame_gui.elements.UIButton(
                    relative_rect=pygame.Rect((0, 50), context_menu_button_size),
                    text='Add edge',
                    manager=self.manager,
                    container=cont
                ),
                pygame_gui.elements.UIButton(
                    relative_rect=pygame.Rect((0, 75), context_menu_button_size),
                    text='Delete edge',
                    manager=self.manager,
                    container=cont
                ),
                pygame_gui.elements.UIButton(
                    relative_rect=pygame.Rect((0, 100), context_menu_button_size),
                    text='Change weight',
                    manager=self.manager,
                    container=cont
                )
            ],
            active=False
        )

    def ui_edit_graph(self) -> None:  # TODO handlers
        menu_button_size = (200, 25)

        def complement_handler(_event):
            pass

        def distribute_handler(_event):
            pass

        def delete_handler(_event):
            pass

        def clear_animation_handler(_event):
            self.stop()
            self.graph_view.reset_animation()

        self.edit_menu = ContextMenu(
            container=(cont := pygame_gui.core.UIContainer(
                pygame.Rect(
                    (-200, -200),
                    (200, 125),
                ),
                manager=self.manager
            )),
            row_height=50,
            items=[
                pygame_gui.elements.UIButton(
                    relative_rect=pygame.Rect((0, 0), menu_button_size),
                    text='Complement',
                    manager=self.manager,
                    container=cont,
                    tool_tip_text='Replace current graph with its complementary graph'
                ).set_handler(complement_handler),
                pygame_gui.elements.UIButton(
                    relative_rect=pygame.Rect((0, 25), menu_button_size),
                    text='Distribute',
                    manager=self.manager,
                    container=cont,
                    tool_tip_text='Find new layout'
                ).set_handler(distribute_handler),
                pygame_gui.elements.UIButton(
                    relative_rect=pygame.Rect((0, 50), menu_button_size),
                    text='Delete graph',
                    manager=self.manager,
                    container=cont,
                    tool_tip_text='Delete current graph'
                ).set_handler(delete_handler),
                pygame_gui.elements.UIButton(
                    relative_rect=pygame.Rect((0, 75), menu_button_size),
                    text='Clear animation',
                    manager=self.manager,
                    container=cont,
                    tool_tip_text='Delete current animation with its coloring, labels and queue'
                ).set_handler(clear_animation_handler)
            ],
            active=False
        )

        def edit_handler(_e):
            self.edit_menu.active = True
            self.open_edit()

        _edit = pygame_gui.elements \
            .UIButton(
                relative_rect=pygame.Rect((self.left_margin, self.top_margin + 195), (100, 50)),
                text='Edit',
                manager=self.manager,
                tool_tip_text='Edit current graph'
            ) \
            .set_handler(edit_handler)

    def ui_choose_algorithm(self) -> None:
        menu_button_size = (200, 25)

        def bfs_handler(_e):
            self.reset_tracker()
            try:
                _ = bfs(self.graph_view.graph, tracker=self.tracker)
                self.play()
            except ArgumentError:
                pass

        def dfs_handler(_e):
            self.reset_tracker()
            try:
                _ = dfs(self.graph_view.graph, tracker=self.tracker)
                self.play()
            except ArgumentError:
                pass

        def binary_bfs_handler(_e):
            self.reset_tracker()
            try:
                _ = binary_bfs(self.graph_view.graph, tracker=self.tracker)
                self.play()
            except ArgumentError:
                pass

        def dijkstra_handler(_e):
            self.reset_tracker()
            try:
                _ = dijkstra(self.graph_view.graph, tracker=self.tracker)
                self.play()
            except ArgumentError:
                pass

        def bellman_ford_handler(_e):
            self.reset_tracker()
            try:
                _ = bellman_ford(self.graph_view.graph, tracker=self.tracker)
                self.play()
            except ArgumentError:
                pass

        def floyd_warshall_handler(_e):
            self.reset_tracker()
            try:
                _ = floyd_warshall(self.graph_view.graph, tracker=self.tracker)
                self.play()
            except ArgumentError:
                pass

        def kruskal_handler(_e):
            self.reset_tracker()
            try:
                _ = kruskal(self.graph_view.graph, tracker=self.tracker)
                self.play()
            except ArgumentError:
                pass

        def prim_handler(_e):
            self.reset_tracker()
            try:
                _ = prim(self.graph_view.graph, tracker=self.tracker)
                self.play()
            except ArgumentError:
                pass

        def articulation_point_handler(_e):
            self.reset_tracker()
            try:
                _ = articulation_point(self.graph_view.graph, tracker=self.tracker)
                self.play()
            except ArgumentError:
                pass

        def bridge_handler(_e):
            self.reset_tracker()
            try:
                _ = bridge(self.graph_view.graph, tracker=self.tracker)
                self.play()
            except ArgumentError:
                pass

        self.choose_algorithm = ContextMenu(
            container=(cont := pygame_gui.core.UIContainer(
                pygame.Rect(
                    (-200, -200),
                    (200, 225),
                ),
                manager=self.manager
            )),
            row_height=50,
            items=[
                pygame_gui.elements.UIButton(
                    relative_rect=pygame.Rect((0, 0), menu_button_size),
                    text='BFS',
                    manager=self.manager,
                    container=cont
                )
                .set_handler(bfs_handler),

                pygame_gui.elements.UIButton(
                    relative_rect=pygame.Rect((0, 25), menu_button_size),
                    text='DFS',
                    manager=self.manager,
                    container=cont
                )
                .set_handler(dfs_handler),

                pygame_gui.elements.UIButton(
                    relative_rect=pygame.Rect((0, 50), menu_button_size),
                    text='Binary BFS',
                    manager=self.manager,
                    container=cont
                )
                .set_handler(binary_bfs_handler),

                pygame_gui.elements.UIButton(
                    relative_rect=pygame.Rect((0, 75), menu_button_size),
                    text='Dijkstra',
                    manager=self.manager,
                    container=cont
                )
                .set_handler(dijkstra_handler),

                pygame_gui.elements.UIButton(
                    relative_rect=pygame.Rect((0, 100), menu_button_size),
                    text='Bellman-Ford',
                    manager=self.manager,
                    container=cont
                )
                .set_handler(bellman_ford_handler),

                pygame_gui.elements.UIButton(
                    relative_rect=pygame.Rect((0, 125), menu_button_size),
                    text='Floyd-Warshall',
                    manager=self.manager,
                    container=cont
                )
                .set_handler(floyd_warshall_handler),

                pygame_gui.elements.UIButton(
                    relative_rect=pygame.Rect((0, 150), menu_button_size),
                    text='Kruskal',
                    manager=self.manager,
                    container=cont
                )
                .set_handler(kruskal_handler),

                pygame_gui.elements.UIButton(
                    relative_rect=pygame.Rect((0, 175), menu_button_size),
                    text='Prim',
                    manager=self.manager,
                    container=cont
                )
                .set_handler(prim_handler),

                pygame_gui.elements.UIButton(
                    relative_rect=pygame.Rect((0, 200), menu_button_size),
                    text='Articulation Point',
                    manager=self.manager,
                    container=cont
                )
                .set_handler(articulation_point_handler),

                pygame_gui.elements.UIButton(
                    relative_rect=pygame.Rect((0, 225), menu_button_size),
                    text='Bridge',
                    manager=self.manager,
                    container=cont
                )
                .set_handler(bridge_handler)
            ],
            active=False
        )

        def choose_algorithm_handler(_e):
            self.choose_algorithm.active = True
            self.open_algorithm()

        _perform_algorithm = pygame_gui.elements \
            .UIButton(
                relative_rect=pygame.Rect((self.left_margin, self.top_margin + 130), (100, 50)),
                text='Perform',
                manager=self.manager,
                tool_tip_text='Perform and visualize an algorithm'
            ) \
            .set_handler(choose_algorithm_handler)

    def ui_save_graph(self) -> None:
        self.save_io = GraphIO(
            input_box=pygame_gui.elements.UITextEntryLine(
                relative_rect=pygame.Rect((self.left_margin + 120, self.top_margin + 65), (1000, 30)),
                initial_text=getcwd(),
                manager=self.manager,
                visible=False
            ),
            active=False
        )

        def save_graph_handler(_e):
            if not self.save_io.input_box.visible and not self.save_io.active:
                self.save_io.input_box.visible = True
                self.save_io.active = True

        _save_graph = pygame_gui.elements \
            .UIButton(
                relative_rect=pygame.Rect((self.left_margin, self.top_margin + 65), (100, 50)),
                text='Save',
                manager=self.manager,
                tool_tip_text='Save graph to file'
            ) \
            .set_handler(save_graph_handler)

    def ui_load_graph(self) -> None:
        self.load_io = GraphIO(
            input_box=pygame_gui.elements.UITextEntryLine(
                relative_rect=pygame.Rect((self.left_margin + 120, self.top_margin), (1000, 30)),
                initial_text=getcwd(),
                manager=self.manager,
                visible=False
            ),
            active=False
        )

        def load_graph_handler(_e):
            if not self.load_io.input_box.visible and not self.save_io.active:
                self.load_io.input_box.visible = True
                self.load_io.active = True

        _load_graph = pygame_gui.elements \
            .UIButton(
                relative_rect=pygame.Rect((self.left_margin, self.top_margin), (100, 50)),
                text='Load',
                manager=self.manager,
                tool_tip_text='Load and display graph from file'
            ) \
            .set_handler(load_graph_handler)

    def run(self) -> None:
        self.running = True
        while self.running:
            time_delta = self.clock.tick(60) / 1000.

            for event in pygame.event.get():
                match event.type:
                    case pygame.QUIT:
                        self.running = False

                    # UI Button handling

                    case pygame_gui.UI_BUTTON_PRESSED \
                            if hasattr(event.ui_element, 'handler'):
                        event.ui_element.handler(event)

                    # File I/O

                    case pygame.KEYDOWN \
                        if event.key == pygame.K_RETURN \
                            and self.load_io.active:

                        self.handle_load_return()

                    case pygame.KEYDOWN \
                        if event.key == pygame.K_RETURN \
                            and self.save_io.active:

                        self.handle_save_return()

                    # Node dragging

                    case pygame.MOUSEBUTTONUP \
                            if self.mouse_attached is not None \
                            and event.button == 1\
                            and not self.dragging_allowed:

                        self.mouse_attached = None
                        self.dragging_allowed = True
                        print('Detached')

                    case pygame.MOUSEBUTTONDOWN \
                        if event.button == 1 \
                            and self.dragging_allowed \
                            and (element := self.draggable_collision(event.pos)) is not None:

                        self.attach_to_mouse(element, event)
                        self.dragging_allowed = False
                        print('Attached')

                    case pygame.MOUSEMOTION if (element := self.mouse_attached) is not None:
                        self.follow_mouse(element, event)

                    case pygame.MOUSEMOTION:
                        self.mouse_x, self.mouse_y = event.pos

                    # Context menu

                    case pygame.MOUSEBUTTONDOWN \
                        if event.button == 3 \
                            and not self.context_menu.active \
                            and self.graph_area.collidepoint(event.pos) \
                            and not self.draggable_collision(event.pos):

                        self.open_context_menu(event.pos)
                        self.context_menu.active = True

                    case pygame.MOUSEBUTTONDOWN:
                        if self.context_menu.active and not self.manager.get_hovering_any_element():
                            self.close_context_menu()
                            self.context_menu.active = False

                        if self.edit_menu.active and not self.manager.get_hovering_any_element():
                            self.close_edit()
                            self.edit_menu.active = False

                        if self.choose_algorithm.active and not self.manager.get_hovering_any_element():
                            self.close_algorithm()
                            self.choose_algorithm.active = False

                self.manager.process_events(event)

            self.manager.update(time_delta)

            self.window_surface.blit(self.background, (0, 0))

            self.draw_graph()
            self.manager.draw_ui(self.window_surface)

            if self.player is not None:
                queues = self.player.animation.current.queues
                if len(queues) > 0:
                    self.draw_queue(queues[0], (0, 40))

            pygame.display.flip()

        self.quit()

    @staticmethod
    def follow_mouse(element: Node, event: pygame.Event) -> None:
        element.position[0], element.position[1] = event.pos

    def attach_to_mouse(self, element: Node, event: pygame.Event) -> None:
        self.mouse_attached = element
        mouse_x, mouse_y = event.pos
        self.mouse_element_offset = Position(
            element.x - mouse_x,
            element.y - mouse_y
        )

    def handle_save_return(self) -> None:
        self.save_io.active = False
        self.save_io.input_box.visible = False
        self.save_graph(self.save_io.input_box.get_text())

    def handle_load_return(self) -> None:
        self.load_io.active = False
        self.load_io.input_box.visible = False
        try:
            self.load_graph(self.load_io.input_box.get_text())
        except FileNotFoundError:
            pass
        finally:
            pass

    def quit(self) -> None:
        if self.player is not None:
            self.player.stop()
        pygame.quit()

    def open_context_menu(self, position: tuple[int, int]) -> None:
        self.context_menu.container.set_position(position)

    def close_context_menu(self) -> None:
        self.open_context_menu((-200, -200))

    def open_edit(self, position: Optional[tuple[int, int]] = None) -> None:
        self.edit_menu.container.set_position(
            position
            if position is not None
            else (self.left_margin + 100, self.top_margin + 195)
        )

    def close_edit(self) -> None:
        self.open_edit((-200, -200))

    def open_algorithm(self, position: Optional[tuple[int, int]] = None) -> None:
        self.choose_algorithm.container.set_position(
            position
            if position is not None
            else (self.left_margin + 100, self.top_margin + 135)
        )

    def close_algorithm(self) -> None:
        self.open_algorithm((-200, 200))

    def draggable_collision(self, position: tuple[int, int]) -> Optional[Node]:
        for node in self.graph_view.nodes.values():
            if np.linalg.norm(node.position - np.array(position)) <= self.NODE_RADIUS:
                return node

    def node_collision(self, position: tuple[int, int]) -> bool:
        if not self.graph_view and position:
            return False

    def draw_queue(self, queue: Any, offset: tuple[int, int]) -> None:
        def truncate(q):
            return q[:self.MAX_QUEUE_LENGTH//2] + ['...'] + q[-self.MAX_QUEUE_LENGTH//2:]

        def draw_element(el, ind, background_color=BLUE, text_color=WHITE):
            active_length = min(len(queue), self.MAX_QUEUE_LENGTH)

            x = (self.WINDOW_SIZE[0] - (self.QUEUE_ELEMENT_SIZE + self.QUEUE_ELEMENT_SPACING) * active_length) / 2 + \
                ind * (self.QUEUE_ELEMENT_SIZE + self.QUEUE_ELEMENT_SPACING) + \
                offset[0]

            y = offset[1] - self.QUEUE_ELEMENT_SIZE / 2

            pygame.draw.rect(
                self.window_surface,
                background_color,
                (x, y, self.QUEUE_ELEMENT_SIZE, self.QUEUE_ELEMENT_SIZE)
            )

            pygame.draw.rect(
                self.window_surface,
                BLACK,
                (x, y, self.QUEUE_ELEMENT_SIZE, self.QUEUE_ELEMENT_SIZE),
                2
            )

            text = pygame.font.Font(None, 24).render(str(el), True, text_color)
            text_rect = text.get_rect(
                center=(
                    x + self.QUEUE_ELEMENT_SIZE / 2,
                    y + self.QUEUE_ELEMENT_SIZE / 2
                )
            )

            self.window_surface.blit(text, text_rect)

        def draw_entry(entry, index, background_color=BLUE, text_color=WHITE):
            match entry:
                case (_priority, number):
                    draw_element(number, index, background_color, text_color)

                case number:
                    draw_element(number, index, background_color, text_color)

        def draw_top_label():
            active_length = min(len(queue), self.MAX_QUEUE_LENGTH)

            text = pygame.font\
                .Font(None, 24)\
                .render('Top', True, BLACK)

            x = (self.WINDOW_SIZE[0] - (self.QUEUE_ELEMENT_SIZE + self.QUEUE_ELEMENT_SPACING) * active_length) / 2 + \
                self.QUEUE_ELEMENT_SIZE / 2 + \
                offset[0]

            y = offset[1] + self.QUEUE_ELEMENT_SIZE

            text_rect = text.get_rect(center=(x, y))

            self.window_surface.blit(text, text_rect)

        if len(queue) == 0:
            return

        view = queue \
            if len(queue) < self.MAX_QUEUE_LENGTH \
            else truncate(queue)

        draw_entry(view[0], 0, GREEN)

        for i, element in enumerate(view[1:], start=1):
            draw_entry(element, i)

        draw_top_label()

    def draw_graph(self) -> None:
        self.draw_background()
        self.draw_edges()
        self.draw_nodes()

    def draw_background(self) -> None:
        pygame.draw.rect(
            self.window_surface,
            '#f5f5f0',
            self.graph_area
        )

    def draw_nodes(self) -> None:
        for node in self.graph_view.nodes.values():
            self.draw_node_interior(node)
            self.draw_node_border(node)
            self.draw_node_label(node)

    def draw_node_label(self, node: Node) -> None:
        label = pygame.font\
            .Font(None, 24)\
            .render(node.label, True, BLACK)

        label_rect = label.get_rect(center=node.as_tuple)

        self.window_surface.blit(label, label_rect)

    def draw_node_border(self, node: Node) -> None:
        pygame.draw.circle(
            self.window_surface,
            '#000000',
            node.as_tuple,
            self.NODE_RADIUS,
            self.NODE_BORDER_WIDTH
        )

    def draw_node_interior(self, node: Node) -> None:
        rect = pygame.draw.circle(
            self.window_surface,
            node.color,
            node.as_tuple,
            self.NODE_RADIUS
        )

        self.draggable.append(rect)

    def draw_edges(self) -> None:
        for start, end in self.graph_view.edges:
            pygame.draw.line(
                self.window_surface,
                self.graph_view.edge_colors[start.vertex, end.vertex],
                start.as_tuple,
                end.as_tuple,
                4
            )

    def load_graph(self, filepath: str) -> None:
        self.graph_view = GraphView.from_file(filepath)

    def save_graph(self, filepath: str) -> None:
        self.graph_view.to_file(filepath)


if __name__ == '__main__':
    app = App()
    app.start()
    app.join()
    print('Shutting down')
