import pygame
import numpy as np
from Render import render, draw_predictions
from sys import exit


pygame.init()
INFO = pygame.display.Info()

# Changeable
WIDTH, HEIGHT = INFO.current_w, INFO.current_h
MAX_FPS = 60

# Internal variables
screen = pygame.display.set_mode((WIDTH, HEIGHT))
FPS_clock = pygame.time.Clock()
is_active = True

# FIXME: Temporary solution
picture_dummy = np.random.randint(0, 255, (28,28)).astype(np.uint8) # Structure efficient for graw shadow storage
ai_dummy = np.random.uniform(0, 1, 10) # 0 - 9 included
# Exit button
exit_rect = pygame.Rect(WIDTH - 50, 10, 40, 40)

# Main loop
if __name__ == '__main__':
    # Initailize drawing constants:
    while is_active:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
                
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Exit button
                if exit_rect.collidepoint(event.pos):
                    pygame.quit()
                    exit()
                # TODO: Drawing
                pass
        
        render(screen, picture_dummy, ai_dummy)

        # FIXME: A bit off-centered vertically to the bottom      
        # Draw predictions bars 
        bar_h = HEIGHT / 3 / 10
        bar_w = WIDTH / 2 / 2
        spacing = bar_h / 10
        start_x = WIDTH / 4 * 3 - (bar_w / 2)
        start_y = HEIGHT / 2 - ((10*bar_h + 9*spacing) / 2)

        font = pygame.font.Font("PixelifySans.ttf", int(bar_h))
        draw_predictions(screen, ai_dummy, start_x, start_y, bar_h, bar_w, spacing, font)

        FPS_clock.tick(MAX_FPS)
        pygame.display.update()
