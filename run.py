import pygame
import numpy as np
import ai
from Render import render, draw_predictions
from Buttons import initialize_buttons
from sys import exit
from os.path import join




pygame.init()
INFO = pygame.display.Info()

# Changeable
WIDTH, HEIGHT = INFO.current_w, INFO.current_h
MAX_FPS = 120

# Internal variables
screen = pygame.display.set_mode((WIDTH, HEIGHT))
FPS_clock = pygame.time.Clock()
is_active = True

current_picture = picture = np.zeros((28, 28), dtype=np.uint8) # Structure efficient for graw shadow storage

# Initialize AI
neural_network = ai.NeuralNetwork([784, 30, 20, 15, 10],
                                  [ai.Linear(), ai.Sigmoid(), ai.ReLU() ,ai.BinaryStep(), ai.Softmax()])

# Main loop
if __name__ == '__main__':
    # Initailize drawing constants:
    picture_size = min(HEIGHT, WIDTH//2)
    pixel_size = picture_size // 28
    # Buttons:
    all_buttons = initialize_buttons(screen, current_picture, neural_network)
    # Bars params:
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
                if pos_x <= picture_size and pos_y <= picture_size:
                    pixel_x = int(pos_x * 28 / picture_size)
                    pixel_y = int(pos_y * 28 / picture_size)

                    # TODO: add shades of grey
                    current_picture[pixel_y][pixel_x] = 255

        # Clear previous frame
        screen.fill((0, 0, 0))
        
        # Update AI predictions in case new pixels were drawn
        predictions = neural_network.forward(current_picture.flatten())

        render(screen, current_picture, pixel_size)
        draw_predictions(screen, predictions, start_x, start_y, bar_h, bar_w, spacing, font)

        # Draw all the buttons
        for b in all_buttons:
            b.draw(screen)

        FPS_clock.tick(MAX_FPS)
        pygame.display.update()
