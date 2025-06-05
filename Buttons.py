import pygame
from sys import exit
from os.path import join

# TODO: add type hints and descriptive doc-strings
class Button:
    def __init__(self, path, function, top_left_position, size=None):
        self._surface = pygame.image.load(path)
        if size:
            self._surface = pygame.transform.scale(self._surface, size)
        self._rectangle = self._surface.get_rect(topleft=top_left_position)
        self._function = function

    def __call__(self, *args, **kwargs):
        return self._function(*args, **kwargs)
    
    def get_rect(self):
        return self._rectangle

    def draw(self, surface):
        surface.blit(self._surface, self._rectangle)


def initialize_buttons(screen):
    """
    Creates Buttons for MNIST number guesser with respect to screen height, and places them with respect to screen width
    returns the Buttons in order:
    exit, import, save, learn
    """
    screen_w, screen_h = screen.get_size()
    button_h = screen_h / 11

    # exit
    def _exit():
        pygame.quit()
        exit()
    exit_b = Button(join("static", "exit.png"), _exit, (screen_w-button_h, 0), (button_h, button_h))

    # import
    def _import():
        # TODO: Select a file, and import either weights into the NeuralNetwork or a picture
        pass
    import_b = Button(join("static", "import.png"), _import, (screen_w-2*button_h, 0), (button_h, button_h))

    # save
    def _save():
        # TODO: Ask whether to save the picture or the weights 
        pass
    save_b = Button(join("static", "save.png"), _save, (screen_w-3*button_h, 0), (button_h, button_h))

   # train
    def _train():
        # TODO: Ask how many cycles of training, learning annealing rate, dropout rate, and k batch sizes, then train the neural network accordingly
        pass
    train_b = Button(join("static", "train.png"), _train, (screen_w-4*button_h, 0), (button_h, button_h))

    return exit_b, import_b, save_b, train_b