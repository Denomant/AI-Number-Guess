import pygame
import numpy as np
from numpy.typing import NDArray


def render(screen: pygame.Surface, picture: NDArray[np.int8], ai_results: NDArray[np.float16]):
    # TODO check the sizes and types of the inputs using validate_array from ai
    if picture.ndim != 2:
        raise ValueError("Picture is expected to be a 2 dimentional ndarray")
    for line in picture:
        if not all((np.issubdtype(j, np.integer)) for j in line):
            raise ValueError("The picture is expected to contain only integers")
        # Separation for the case the contents of picture does not support comparison
        if not all((j <= 255) for j in line):
            raise ValueError("The picture is expected to contain only integers smaller than 255")

    width, height = screen.get_size()
    width = 1 * height # Adjust factor if neccesary

    pixel_w, pixel_h = width / len(picture[0]), len(picture)
    pixel_rect = pygame.Rect(0, 0, pixel_w, pixel_h)
    for line in picture:
        for pixel in line:
            # Draw pixel
            pygame.draw.rect(screen, (pixel, pixel, pixel), pixel_rect)
            # Update next pixel location
            pixel_rect.left += pixel_w
        pixel_rect.bottom += pixel_h
        pixel_rect.left = 0
        

def draw_predictions(surface, predictions, start_x, start_y, bar_height=20, bar_width=200, spacing=10, font=None):
    """
    Displays 10 predictions in percentage form (from 0-1) 

    surface: pygame.Surface where the predictions are going to get drawn
    predictions: list or a ndarray of 10 floats (0–1)
    start_x: The x of the top left corner of the bar at the very top
    start_y: The y of the yop left corner of the bar at the very top
    bar_height: The height of a single bar
    bar_width: The width of a single bar
    spacing: The distance between the bars
    font: pygame.font.Font object for text
    """
    # TODO: Check the rest of the parametrs

    max_conf = max(predictions)
    for i, prob in enumerate(predictions):
        percent = prob * 100
        filled = int(prob * bar_width)

        # Green color for the most condident prediction, blue for others.
        color = (50, 200, 50) if prob == max_conf else (100, 100, 255)

        y = start_y + i * (bar_height + spacing)

        # The background of rach bar
        pygame.draw.rect(surface, (220, 220, 220), (start_x, y, bar_width, bar_height))

        # The filled part of each bar
        pygame.draw.rect(surface, color, (start_x, y, filled, bar_height))

        # Drawing the numbers a labels using font
        if font:
            label = font.render(f'{i}', True, (255, 255, 255))
            surface.blit(label, (start_x - 25, y))
            
            percent_text = font.render(f'{percent:.2f}%', True, (255, 255, 255))
            surface.blit(percent_text, (start_x + bar_width + 10, y))