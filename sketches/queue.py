import pygame
import sys

# Initialize Pygame
pygame.init()

# Set the width and height of the screen
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 400
SCREEN_SIZE = (SCREEN_WIDTH, SCREEN_HEIGHT)
SCREEN = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption("Queue Visualization")

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Define the queue and its maximum size
queue = []
MAX_QUEUE_SIZE = 10

# Define the size and spacing of queue elements
ELEMENT_SIZE = 30
ELEMENT_SPACING = 10


def truncate(q):
    return q[:MAX_QUEUE_SIZE//2] + ['...'] + q[-MAX_QUEUE_SIZE//2:]


# Function to draw the queue elements
def draw_queue():
    if len(queue) == 0:
        return

    queue_view = queue if len(queue) < MAX_QUEUE_SIZE else truncate(queue)

    draw_element(queue_view[0], 0, GREEN)

    for i, element in enumerate(queue_view[1:], start=1):
        draw_element(element, i)

    draw_top()


def draw_top():
    text = pygame.font.Font(None, 24).render('Top', True, BLACK)
    text_rect = text.get_rect(center=(
        (SCREEN_WIDTH - (ELEMENT_SIZE + ELEMENT_SPACING) * min(len(queue), MAX_QUEUE_SIZE)) / 2 + ELEMENT_SIZE / 2,
        SCREEN_HEIGHT / 2 + ELEMENT_SIZE
    ))
    SCREEN.blit(text, text_rect)


def draw_element(element, i, background_color=BLUE, text_color=WHITE):
    x = (SCREEN_WIDTH - (ELEMENT_SIZE + ELEMENT_SPACING) * min(len(queue), MAX_QUEUE_SIZE)) / 2 + i * (ELEMENT_SIZE + ELEMENT_SPACING)
    y = SCREEN_HEIGHT / 2 - ELEMENT_SIZE / 2
    pygame.draw.rect(SCREEN, background_color, (x, y, ELEMENT_SIZE, ELEMENT_SIZE))
    pygame.draw.rect(SCREEN, BLACK, (x, y, ELEMENT_SIZE, ELEMENT_SIZE), 2)
    font = pygame.font.Font(None, 24)
    text = font.render(str(element), True, text_color)
    text_rect = text.get_rect(center=(x + ELEMENT_SIZE / 2, y + ELEMENT_SIZE / 2))
    SCREEN.blit(text, text_rect)


# Main game loop
while True:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            # Add an element to the queue when space bar is pressed
            if event.key == pygame.K_SPACE:
                queue.append(len(queue) + 1)
            # Remove the first element from the queue when backspace is pressed
            elif event.key == pygame.K_BACKSPACE:
                if queue:
                    queue.pop(0)

    # Clear the screen
    SCREEN.fill(WHITE)

    # Draw the queue
    draw_queue()

    # Update the screen
    pygame.display.flip()
