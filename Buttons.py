import pygame
import numpy as np
import pickle
import cv2
from sys import exit
from os.path import join, exists

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


def Popup(screen, left, top, width, height, *args, font_dir=None, title="Popup"):
    """
    creates a popup on screen of size width x height, the way that the top left corner is at (left, top)
    then creates len(args) fields to fill, each with args[i] default value (promt for user)
    """
    if len(args) == 0:
        raise ValueError("At least one field is required")
    FPS_clock = pygame.time.Clock()
    active = True
    
    # Reserve bottom 80% of popup for input fields
    field_area_height = int(height * 0.8)
    top_area_height = height - field_area_height
    top_dead_area = int(height * 0.05)
    max_box_height = 50
    min_spacing = 10

    popup_rect = pygame.Rect(left, top, width, height)
    input_boxes = []
    placeholders = args
    input_texts = [''] * len(args)
    selected_box = 0

    # Setup input boxes (fields)
    # Determine how much space I can use per row (box + spacing)
    rows = len(args)
    ideal_total_height = rows * (max_box_height + min_spacing)

    if ideal_total_height <= field_area_height:
        box_height = max_box_height
        spacing = min_spacing
    else:
        # Scale down to fit within 80% block
        box_height = field_area_height // (rows + 1)
        spacing = box_height // 5
        box_height = box_height - spacing  # ensure it fits exactly

    start_y = top + top_area_height  # push fields into the bottom 80%
    offset_x = 20
    for i in range(len(args)):
        input_boxes.append(pygame.Rect(left+offset_x, start_y+i*(box_height+spacing), width-2*offset_x, box_height))

    # OK and Cancel buttons
    button_w, button_h = int(width / 6), int(top_area_height - 2*top_dead_area)
    ok_rect = pygame.Rect(left + width - (2 * button_w + 2 * offset_x), top + top_dead_area, button_w, button_h)
    cancel_rect = pygame.Rect(left + width - (button_w + offset_x), top + top_dead_area, button_w, button_h)

    # fonts matching box sizes
    font_size = box_height - 4
    font = pygame.font.Font(font_dir, font_size)
    title_font = pygame.font.Font(font_dir, top_area_height-2*top_dead_area)
    # Match button_font so the word "Cancel" fits exactly in 90% of button_w
    test_button_font = pygame.font.Font(font_dir, 150)
    test_text_width, test_text_height = test_button_font.size("Cancel")
    scale = (button_w * 0.9) / test_text_width
    final_size = int(150 * scale)
    del test_text_width, test_button_font, test_text_height, scale
    button_font = pygame.font.Font(font_dir, final_size)
    

    while active:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            
            if event.type == pygame.KEYDOWN:
                match event.key:
                    # Close Popup when ESC clicked
                    case pygame.K_ESCAPE:
                        return None
                    # Rotate box selection when TAB or ENTER is clicked
                    case pygame.K_TAB | pygame.K_RETURN | pygame.K_KP_ENTER:
                        selected_box = (selected_box + 1) % len(input_texts)
                    # Remove one symbol from selected box when backspace is clicked
                    case pygame.K_BACKSPACE:
                        input_texts[selected_box] = input_texts[selected_box][:-1:]
                    # Else add the corresponding symbol to selected box if its printable
                    case _:
                        if event.unicode.isprintable():
                            input_texts[selected_box] += event.unicode
                
            if event.type == pygame.MOUSEBUTTONDOWN:
                # If one of the boxes clicked, select this box
                for i, rect in enumerate(input_boxes):
                    if rect.collidepoint(event.pos):
                        selected_box = i

                # If okay is clicked, return current texts
                if ok_rect.collidepoint(event.pos):
                    return input_texts
                # if Cancel is clicked, return None
                if cancel_rect.collidepoint(event.pos):
                    return None
                        
        # Popup window
        pygame.draw.rect(screen, (0, 0, 0), popup_rect)
        pygame.draw.rect(screen, (255, 255, 255), popup_rect, 2) # White border

        # Draw title
        title_surf = title_font.render(title, True, (255, 255, 255))
        title_rect = title_surf.get_rect()
        title_rect.midleft = (left + offset_x, top + top_area_height // 2)
        screen.blit(title_surf, title_rect)

        # Draw input boxes
        for i, rect in enumerate(input_boxes):
            color = (0, 0, 255) if i == selected_box else (255, 255, 255)
            pygame.draw.rect(screen, color, rect, 2)
            if input_texts[i] == '':
                text_surf = font.render(placeholders[i], True, (200, 200, 200))
            else:
                text_surf = font.render(input_texts[i], True, (255, 255, 255))
            screen.blit(text_surf, (rect.x + 5, rect.y + 5))

        # Draw buttons
        for rect, label, color in [(ok_rect, "OK", (0, 255, 0)), (cancel_rect, "Cancel", (255, 0, 0))]:
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (255, 255, 255), rect, 2)
            label_surf = button_font.render(label, True, (255, 255, 255))
            screen.blit(label_surf, label_surf.get_rect(center=rect.center))

        pygame.display.flip()
        FPS_clock.tick(30)


def initialize_buttons(screen, picture, ai):
    """
    Creates Buttons for MNIST number guesser with respect to screen height, and places them with respect to screen width
    returns the Buttons in order:
    exit, import, save, learn, clear
    """
    screen_w, screen_h = screen.get_size()

    button_h = screen_h / 11

    get_static = lambda x: join("static", x)

    popup_size_ratio = 0.8
    popup_w, popup_h = popup_size_ratio * screen_w, popup_size_ratio * screen_h
    popup_left, popup_top = screen_w // 2 - 0.5 * popup_w, screen_h // 2 - 0.5 * popup_h
    popup_config = (int(popup_left), int(popup_top), int(popup_w), int(popup_h))

    # TODO: Add descriptive doc strings
    # exit
    def _exit():
        pygame.quit()
        exit()
    exit_b = Button(join("static", "exit.png"), _exit, (screen_w-button_h, 0), (button_h, button_h))

    # import
    def _import():
        # TODO: Select a file, and import either weights into the NeuralNetwork or a picture
        fields = ["Are you trying to load a picture, or a trained neural network?",
                  "Type the file name you are trying to load"]
        title = "Load Options"
        
        while True:
            options = Popup(screen, *popup_config, *fields, title=title)

            # If cancelled, just leave
            if options is None:
                return None

            # If at least one field was empty, or filled incoorectly - try again with an error message in the title
            if '' in options:
                title = "Fail: Some Fields Were Empty"
                continue

            save_type, save_file = options

            if save_type.lower() not in ("picture", "neural network"):
                title = "Fail: No Such Save Option"
                continue

            extention = ".png" if save_type.lower() == "picture" else ".pkl"
            path = save_file + extention

            if not exists(path):
                title = "Fail: No Such File Exist"
                continue

            # If all fields are filled correctly
            # If a picure
            if extention == ".png":
                image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    title = "Fail: To Load Image"
                    continue

                if image.shape != (28, 28):
                    title = "Fail: Image Is Not 28x28"
                    continue

                np.copyto(picture, image)
                return None

            elif extention == ".pkl":
                # TODO: Load Neural Network using pickle
                return None
            else:
                title = "Fail: Unknown Error"
                continue

    import_b = Button(join("static", "import.png"), _import, (screen_w-2*button_h, 0), (button_h, button_h))

    # save
    def _save():
        fields = ["Would you want to save the current picture, or the trained weights?",
                  "How would you want to name the saved file? (no extention needed)"]
        options = Popup(screen, *popup_config, *fields, title="Save Options")
    save_b = Button(join("static", "save.png"), _save, (screen_w-3*button_h, 0), (button_h, button_h))

    # train
    def _train():
        # TODO: Ask how many cycles of training, learning annealing rate, dropout rate, and k batch sizes, then train the neural network accordingly
        pass
    train_b = Button(join("static", "train.png"), _train, (screen_w-4*button_h, 0), (button_h, button_h))

    # clear
    def _clear():
        picture.fill(0)
        

    clear_b = Button(join("static", "clear.png"), _clear, (screen_w-5*button_h, 0), (button_h, button_h))
    
    return exit_b, import_b, save_b, train_b, clear_b