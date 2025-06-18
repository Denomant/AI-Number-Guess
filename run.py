import pygame
import numpy as np
from Render import render, draw_predictions
from Buttons import initialize_buttons
from sys import exit
from os.path import join



pygame.init()
INFO = pygame.display.Info()

# Changeable
WIDTH, HEIGHT = INFO.current_w, INFO.current_h
MAX_FPS = 60

# Internal variables
screen = pygame.display.set_mode((WIDTH, HEIGHT))
FPS_clock = pygame.time.Clock()
is_active = True

current_picture = picture = np.zeros((28, 28), dtype=np.uint8) # Structure efficient for graw shadow storage

# FIXME: Temporary solution
ai_dummy = np.random.uniform(0, 1, 10) # 0 - 9 included

# Main loop
if __name__ == '__main__':
    # Initailize drawing constants:
    picture_size = min(HEIGHT, WIDTH//2)
    pixel_size = picture_size // 28
    # Buttons:
    all_buttons = initialize_buttons(screen)
    # Bars params:
    # FIXME: A bit off-centered vertically to the bottom
    bar_h = HEIGHT / 3 / 10
    bar_w = WIDTH / 2 / 2
    spacing = bar_h / 10
    start_x = WIDTH / 4 * 3 - (bar_w / 2)
    start_y = HEIGHT / 2 - ((10*bar_h + 9*spacing) / 2)

    font = pygame.font.Font(join("static", "PixelifySans.ttf"), int(bar_h))
    while is_active:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
                
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Button activations
                pos = event.pos
                for b in all_buttons:
                    if b.get_rect().collidepoint(pos):
                        b()

            # Draw while mouse is held down
            if pygame.mouse.get_pressed()[0]:  # Left mouse button is held
                pos_x, pos_y = pygame.mouse.get_pos()
                if pos_x <= picture_size and pos_y <= picture_size * 28:
                    pixel_x = int(pos_x * 28 / picture_size)
                    pixel_y = int(pos_y * 28 / picture_size)
                    current_picture[pixel_y][pixel_x] = 255

        render(screen, current_picture, pixel_size)
        draw_predictions(screen, ai_dummy, start_x, start_y, bar_h, bar_w, spacing, font)

        # Draw all the buttons
        for b in all_buttons:
            b.draw(screen)

        FPS_clock.tick(MAX_FPS)
        pygame.display.update()
