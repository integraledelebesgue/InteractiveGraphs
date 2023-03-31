import threading
from dataclasses import dataclass

import pygame
import pygame_gui

from src.library.graph.prototype_graph_view import PrototypeGraphView

window_size = (1600, 1200)


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
    position: Position
    background: pygame.Rect
    row_height: int
    items: list[pygame_gui.elements.UIButton]


class App(threading.Thread):
    def __init__(self):
        super().__init__()
        pygame.init()
        pygame.display.set_caption('Interactive graphs')

        self.window_surface = pygame.display \
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

        self.graph_view = PrototypeGraphView()

        self.graph_area_corner = (self.left_margin + 120, self.top_margin)
        self.graph_area_size = (1445, 1170)

        self.graph_area = pygame.Rect(
            self.graph_area_corner,
            self.graph_area_size
        )

        self.draggable = []
        self.dragging_allowed = True
        self.mouse_attached = None
        self.mouse_element_offset = None

        self.mouse_x = 0
        self.mouse_y = 0

        self.context_menu = ContextMenu(
            position=Position(-100, -100),
            background=(back := pygame.Rect((-100, -100), (100, 150))),
            row_height=50,
            items=[
                pygame_gui.elements.UIButton(
                    relative_rect=pygame.Rect((back.x, back.y), (100, 50)),
                    text='Button1',
                    manager=self.manager,
                ),
                pygame_gui.elements.UIButton(
                    relative_rect=pygame.Rect((back.x, back.y + 50), (100, 50)),
                    text='Button2',
                    manager=self.manager,
                ),
                pygame_gui.elements.UIButton(
                    relative_rect=pygame.Rect((back.x, back.y + 100), (100, 50)),
                    text='Button1',
                    manager=self.manager,
                )
            ]
        )

        self.context_menu_active = False

        self.init_ui()

    def init_ui(self):
        def new_graph_button_handler(_e):
            print('Hello')

        _new_graph_button = pygame_gui.elements \
            .UIButton(
                relative_rect=pygame.Rect((self.left_margin, self.top_margin), (100, 50)),
                text='Hello',
                manager=self.manager,
                tool_tip_text='Print \'Hello\' in console'
            )\
            .set_handler(new_graph_button_handler)

    def run(self):
        self.running = True
        while self.running:
            time_delta = self.clock.tick(60) / 1000.

            for event in pygame.event.get():
                match event.type:
                    case pygame.QUIT:
                        self.running = False

                    case pygame_gui.UI_BUTTON_PRESSED:
                        try: event.ui_element.handler(event)
                        finally: pass

                    # Node dragging part:

                    case pygame.MOUSEBUTTONDOWN \
                        if event.button == 1 \
                            and self.dragging_allowed \
                            and (element := self.draggable_collision(event.pos)):

                        self.mouse_attached = element

                        mouse_x, mouse_y = event.pos
                        self.mouse_element_offset = Position(
                            element.x - mouse_x,
                            element.y - mouse_y
                        )

                    case pygame.MOUSEBUTTONUP \
                            if self.mouse_attached and event.button == 1:
                        self.mouse_attached = None

                    case pygame.MOUSEMOTION if (element := self.mouse_attached):
                        mouse_x, mouse_y = event.pos
                        element.x = mouse_x + self.mouse_element_offset.x
                        element.y = mouse_y + self.mouse_element_offset.y

                    case pygame.MOUSEMOTION:
                        self.mouse_x, self.mouse_y = event.pos

                    # Context menu part:

                    case pygame.MOUSEBUTTONDOWN \
                        if event.button == 3 \
                            and not self.context_menu_active\
                            and self.graph_area.collidepoint(event.pos) \
                            and not self.node_collision(event.pos):

                        self.open_context_menu(event.pos)
                        self.context_menu_active = True

                    case pygame.MOUSEBUTTONDOWN \
                            if self.context_menu_active:
                        self.close_context_menu()
                        self.context_menu_active = False

                self.manager.process_events(event)

            self.manager.update(time_delta)

            self.window_surface.blit(self.background, (0, 0))
            self.draw_graph()
            self.manager.draw_ui(self.window_surface)
            pygame.display.flip()

        self.quit()

    @staticmethod
    def quit():
        pygame.quit()
        quit()

    def open_context_menu(self, position):
        print(f"Opening context menu on {position}")
        self.context_menu.position = position
        self.context_menu.background.x, self.context_menu.background.y = position
        for i, item in enumerate(self.context_menu.items):
            item.relative_rect.y = position[1] + i * self.context_menu.row_height
            item.relative_rect.x = position[0]

    def close_context_menu(self):
        self.open_context_menu((-100, -100))

    def draggable_collision(self, position):
        return next(
            filter(
                lambda el: el.collidepoint(position),
                self.draggable
            ),
            None
        )

    def node_collision(self, position):
        if not self.graph_view and position:
            return False

    def draw_graph(self):
        pygame.draw.rect(
            self.window_surface,
            '#f5f5f0',
            self.graph_area
        )

        for start, end in self.graph_view.edges:
            pygame.draw.line(
                self.window_surface,
                '#000000',
                start.shift(self.graph_area_corner),
                end.shift(self.graph_area_corner),
                4
            )

        for node in self.graph_view.nodes:
            pygame.draw.circle(
                self.window_surface,
                '#0086b3',
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


if __name__ == '__main__':
    app = App()
    app.run()
    app.join()
