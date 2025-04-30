import pygame
import numpy as np
from Render import render
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
picture_dummy = np.zeros((28,28), dtype=np.int8) # Structure efficient for graw shadow storage
ai_dummy = list(range(10)) # 0 - 9 included

# Main loop
if __name__ == '__main__':
    while is_active:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
                
            if event.type == pygame.MOUSEBUTTONDOWN:
                # TODO: Drawing
                pass

        render(screen, picture_dummy, ai_dummy)
        FPS_clock.tick(MAX_FPS)
        pygame.display.update()