import pygame
import sys

# Initialize Pygame
pygame.init()

# Define colors
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Set up the drawing window
screen = pygame.display.set_mode([500, 500])

# Function to draw the intersection
def draw_intersection():
    screen.fill(BLACK)  # Clear screen with black
    pygame.draw.line(screen, GREEN, (250, 0), (250, 500), 5)  # Vertical line
    pygame.draw.line(screen, GREEN, (0, 250), (500, 250), 5)  # Horizontal line

# Run until the user asks to quit
def run_simulation():
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        draw_intersection()
        pygame.display.flip()

    pygame.quit()
    sys.exit()
