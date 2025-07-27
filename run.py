import pygame
import numpy as np
import ai
import cv2
from Render import render, draw_predictions, paint_brush
from Buttons import initialize_buttons
from sys import exit
from os.path import join, isfile, basename
from os import walk

# Create custom DataPiece subclass to dynamically load images
class MNISTDataPiece(ai.DataPiece):
    """
    Loads a grayscale 28x28 MNIST image from disk when get_data() is called.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        IOError: If the image can't be loaded.
        ValueError: If the image is not 28x28 pixels.
    """
    def __init__(self, image_path):
        """
        Checks that given file exists.

        Parameters:
        - image_path (str): Expected to contain a full path to the image file (relative or absolute).
        """
        if not isfile(image_path):
            raise FileNotFoundError(f"{image_path} does not exist")
        self._path = image_path

    def get_data(self):
        """
        Loads the image from self._path, validates MNIST shape (28x28), and returns it as a flattened grayscale of shape (784,).

        Returns:
        - image (np.ndarray): Flattened grayscale version of self._path.
        """
        image = cv2.imread(self._path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise IOError(f"Could Not Load {self._path}. The file might be corrupted or not an image.")

        if image.shape != (28, 28):
            raise ValueError(f"{self._path} has invalid dimensions. It's {image.shape} when (28, 28) expected.")

        # Return flatten version to match NeuralNetwork input layer shape
        return image.flatten()


def initialize_MNISTDataPieces(root_directory : str=join("dataset_decoded", "train")):
    data = {}

    for dirpath, dirnames, filenames in walk(root_directory):
        if dirpath == root_directory:
            continue # Skip the root directory itself

        # Calculate the label just once for each sub directory
        label = basename(dirpath)
        # Edge cases
        if not label.isdigit():
                raise ValueError(f"Failed to parse label from directory path '{dirpath}'. Expected a numeric label after '{root_directory}', but got '{label}'.")
        label = int(label)
        if label > 9 or label < 0:
            raise ValueError(f"Invalid label: {label}. Expected a number between 0-9.")

        # Convert label to hot-spot format to match Neural network last layer structure
        label = np.array([0 if i!=label else 1 for i in range(10)], dtype=np.int8)

        for filename in filenames:
            # Map the image file (wrapped in MNISTDataPiece object) to its one-hot encoded label
            data[MNISTDataPiece(join(dirpath, filename))] = label

    return data


pygame.init()
INFO = pygame.display.Info()

# Changeable
WIDTH, HEIGHT = INFO.current_w, INFO.current_h
MAX_FPS = 120
brush_radius = 80

# Internal variables
screen = pygame.display.set_mode((WIDTH, HEIGHT))
FPS_clock = pygame.time.Clock()
is_active = True

current_picture = np.zeros((28, 28), dtype=np.uint8) # uint8 does not support negatives but efficient for gray-scales

# Initialize AI, and predicions list
neural_network = ai.NeuralNetwork([784, 30, 20, 15, 10],
                                  [ai.Linear(), ai.Sigmoid(), ai.ReLU() ,ai.BinaryStep(), ai.Softmax()])
predictions = neural_network.forward(current_picture.flatten())

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

# Main loop
if __name__ == '__main__':
    train_data = initialize_MNISTDataPieces()

    while is_active:
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
                
            if event.type == pygame.MOUSEBUTTONDOWN:
                match event.button:
                    case 1: # LMB
                        # Check custom buttons
                        pos = event.pos
                        for b in all_buttons:
                            if b.get_rect().collidepoint(pos):
                                b()
                        
                        # In case the image or AI itself changed, predictaions should be recalculated
                        predictions = neural_network.forward(current_picture.flatten())

                        # Drawing implemented when LMB is held, not when clicked

                    # Decrease or increase brush size depending on the direction of the mouse scroll
                    case 4: # Scroll up
                        brush_radius += 10 
                        brush_radius = round(brush_radius, -1) # Round to the nearest 10, if last size was pixel_size

                    case 5: # Scroll down
                        brush_radius -= 10
                        brush_radius = max(brush_radius, int(pixel_size * 1.5)) # prevent negative numbers, and force brush to always exist

        # Draw while mouse is held down, regardless of event loop
        if pygame.mouse.get_pressed()[0]:  # Left mouse button is held
            pos_x, pos_y = mouse_pos
            if pos_x < picture_size and pos_y < picture_size:
                paint_brush(current_picture, pos_x, pos_y, pixel_size, picture_size, brush_radius)

                # Update AI predictions only right after new pixels were drawn
                predictions = neural_network.forward(current_picture.flatten())

        # Clear previous frame
        screen.fill((0, 0, 0))

        # Main render: image and predictions 
        render(screen, current_picture, pixel_size)
        draw_predictions(screen, predictions, start_x, start_y, bar_h, bar_w, spacing, font)

        # Draw the current brush area, if its inside the picture
        if mouse_pos[0] < picture_size and mouse_pos[1] < picture_size:
            pygame.draw.circle(screen, (255, 255, 255), mouse_pos, brush_radius, 2)

        # Draw all the buttons
        for b in all_buttons:
            b.draw(screen)

        FPS_clock.tick(MAX_FPS)
        pygame.display.update()
