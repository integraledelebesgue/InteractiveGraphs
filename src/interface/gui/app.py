import threading
from dataclasses import dataclass
from os import getcwd
from typing import Any

import numpy as np
import pygame
import pygame_gui

from src.library.algorithms.spanning_trees.kruskal import kruskal
from src.library.algorithms.traversals.bfs import bfs
from src.library.algorithms.traversals.dfs import dfs
from src.library.graph.graph import Graph, GraphView, Animation, Tracker

window_size = (1600, 1200)

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
    def __init__(self):
        super().__init__()
        pygame.init()
        pygame.display.set_caption('Interactive graphs')

        self.window_surface = pygame.display\
            .set_mode(
                window_size,
                pygame.RESIZABLE
            )

        self.background = pygame.Surface(window_size)
        self.background.fill(pygame.Color('#ffffff'))

        self.manager = pygame_gui.UIManager(window_size)
        self.clock = pygame.time.Clock()
        self.running = False

        self.left_margin = 15
        self.right_margin = 15
        self.top_margin = 15
        self.bottom_margin = 15

        self.graph_area_corner = (self.left_margin + 120, self.top_margin)
        self.graph_area_size = (1445, 1170)

        self.graph_area = pygame.Rect(
            self.graph_area_corner,
            self.graph_area_size
        )

        self.graph_view = test_graph.view(self.graph_area_size)
        # self.graph_view.distribute(50)

        self.tracker = Tracker()

        kruskal(self.graph_view.graph, tracker=self.tracker)

        self.player = self.tracker\
            .as_animation_of(self.graph_view)\
            .player()\
            .set_delay(1.0)

        self.player.start()

        self.draggable = []
        self.dragging_allowed = True
        self.mouse_attached = None
        self.mouse_element_offset = None

        self.mouse_x = 0
        self.mouse_y = 0

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

        _load_graph = pygame_gui.elements\
            .UIButton(
                relative_rect=pygame.Rect((self.left_margin, self.top_margin), (100, 50)),
                text='Load',
                manager=self.manager,
                tool_tip_text='Load new graph from file and display in editor'
            )\
            .set_handler(load_graph_handler)

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

        _save_graph = pygame_gui.elements\
            .UIButton(
                relative_rect=pygame.Rect((self.left_margin, self.top_margin + 65), (100, 50)),
                text='Save',
                manager=self.manager,
                tool_tip_text='Save current graph to file'
            )\
            .set_handler(save_graph_handler)

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

    def run(self) -> None:
        self.running = True
        while self.running:
            time_delta = self.clock.tick(60) / 1000.

            for event in pygame.event.get():
                match event.type:
                    case pygame.QUIT:
                        self.running = False

                    # UI Button handling

                    case pygame_gui.UI_BUTTON_PRESSED\
                            if hasattr(event.ui_element, 'handler'):
                        event.ui_element.handler(event)

                    # File I/O

                    case pygame.KEYDOWN\
                        if event.key == pygame.K_RETURN\
                            and self.load_io.active:

                        self.handle_load_return()

                    case pygame.KEYDOWN\
                        if event.key == pygame.K_RETURN\
                           and self.save_io.active:

                        self.handle_save_return()

                    # Node dragging

                    case pygame.MOUSEBUTTONDOWN\
                        if event.button == 1\
                            and self.dragging_allowed\
                            and (element := self.draggable_collision(event.pos)):

                        self.attach_to_mouse(element, event)

                    case pygame.MOUSEBUTTONUP\
                            if self.mouse_attached and event.button == 1:
                        self.mouse_attached = None

                    case pygame.MOUSEMOTION if (element := self.mouse_attached):
                        self.follow_mouse(element, event)

                    case pygame.MOUSEMOTION:
                        self.mouse_x, self.mouse_y = event.pos

                    # Context menu

                    case pygame.MOUSEBUTTONDOWN\
                        if event.button == 3\
                            and not self.context_menu.active\
                            and self.graph_area.collidepoint(event.pos)\
                            and not self.node_collision(event.pos):

                        self.open_context_menu(event.pos)
                        self.context_menu.active = True

                    case pygame.MOUSEBUTTONDOWN\
                        if self.context_menu.active:

                        self.close_context_menu()
                        self.context_menu.active = False

                self.manager.process_events(event)

            self.manager.update(time_delta)

            self.window_surface.blit(self.background, (0, 0))
            self.draw_graph()
            self.manager.draw_ui(self.window_surface)
            pygame.display.flip()

        self.quit()

    def follow_mouse(self, element, event):
        mouse_x, mouse_y = event.pos
        element.x = mouse_x + self.mouse_element_offset.x
        element.y = mouse_y + self.mouse_element_offset.y

    def attach_to_mouse(self, element, event):
        self.mouse_attached = element
        mouse_x, mouse_y = event.pos
        self.mouse_element_offset = Position(
            element.x - mouse_x,
            element.y - mouse_y
        )

    def handle_save_return(self):
        self.save_io.active = False
        self.save_io.input_box.visible = False
        self.save_graph(self.save_io.input_box.get_text())

    def handle_load_return(self):
        self.load_io.active = False
        self.load_io.input_box.visible = False
        try:
            self.load_graph(self.load_io.input_box.get_text())
        except FileNotFoundError:
            pass
        finally:
            pass

    def quit(self) -> None:
        self.player.stop()
        pygame.quit()

    def open_context_menu(self, position) -> None:
        self.context_menu.container.set_position(position)

    def close_context_menu(self) -> None:
        self.open_context_menu((-200, -200))

    def draggable_collision(self, position) -> Any:
        return next(
            filter(
                lambda el: el.collidepoint(position),
                self.draggable
            ),
            None
        )

    def node_collision(self, position) -> bool:
        if not self.graph_view and position:
            return False

    def draw_graph(self) -> None:
        self.draw_background()
        self.draw_edges()
        self.draw_nodes()

    def draw_background(self):
        pygame.draw.rect(
            self.window_surface,
            '#f5f5f0',
            self.graph_area
        )

    def draw_nodes(self):
        for node in self.graph_view.nodes.values():
            pygame.draw.circle(
                self.window_surface,
                node.color,
                node.shift(self.graph_area_corner),
                15
            )

            pygame.draw.circle(
                self.window_surface,
                '#000000',
                node.shift(self.graph_area_corner),
                15,
                3
            )

    def draw_edges(self):
        for start, end in self.graph_view.edges:
            pygame.draw.line(
                self.window_surface,
                self.graph_view.edge_colors[start.vertex, end.vertex],
                start.shift(self.graph_area_corner),
                end.shift(self.graph_area_corner),
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
