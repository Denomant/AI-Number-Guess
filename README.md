# AI-Number-Guess
## Table of Contents
- [Project Overview](#project-overview)
- [What's Next](#whats-next)
- [Test Results](#test-results)
- [Try Yourself](#try-yourself)
	- [Getting Started](#getting-started)
	- [Logic Test Benchmarks](#logic-test-benchmarks)
	- [Pygame Paint Application](#pygame-paint-application)
- [Technical Detail](#technical-detail)
    - [Files and Classes](#files-and-classes)
	- [Critical Flaws](#critical-flaws)
- [Download MNIST](#download-mnist)

## Project Overview
The project began as an ambitious attempt to design a fully modular, fully customizable neural network module, written entirely from
scratch without relying on ready-to-go frameworks or libraries such as scikit-learn, TensorFlow, PyTorch, autograd, etc. The idea
was to build the entire system of the forward pass, backpropagation, and gradient descent fully by hand, and also make it flexible
so that every aspect of the network could be configured: any amount of layers, any size of layers, any activation functions (including
user-defined ones, such as mixed-activation layers, or experimental modern architectures).  

Secondly, I wanted to showcase my AI and integrate it into a small Pygame-based paint-like interface, where the user could draw a
digit and the AI would classify it in real time using the MNIST dataset. While the network demonstrated nearly flawless
performance on all internal logic benchmarks, scaling it to the MNIST dataset revealed weaknesses in the architecture’s initial design. 

## What's Next
While the project achieved its main goals — a working and fully configurable, any-size, any-purpose neural network module — 
the internal structure and API design were not well thought out in advance. As a result, when the codebase grew, these design flaws
made further development, maintenance, and application too complex and practically impossible. 

Currently, the project is abandoned / archived because its architectural limitations could not be fixed with endless patching, and would require
a complete redesign from the ground up prioritizing proper software architecture and clearer separation of concerns. I may return to
this project in the future to attempt a proper redesign, but not in the foreseeable future.

## Test Results
`TESTS.py:`

| Type | Description                  | Average Loss Before Learning | Average Loss After Learning | Improvement |
|-----------|------------------------------|-----------------------------|----------------------------|-------------|
| AND       | a AND b                       | 0.61733                     | 0.041405                   | 93.29 %     |
| OR        | a OR b                        | 0.626239                    | 0.036395                   | 94.19 %     |
| XOR       | a XOR b                       | 0.49816                     | 0.053286                   | 89.30 %     |
| LOGIC     | (a AND NOT b) XOR (c AND d)  | 0.406882                    | 0.016805                   | 95.87 %      |

Across all tasks, the AI learned effectively, reducing average loss by 89–96%, demonstrating the core network — forward pass,
backpropagation, and gradient descent — were implemented correctly despite architectural complexities.


## Try Yourself
### Getting Started
1. Make sure you have [Python 3](https://www.python.org/) and [git](https://git-scm.com/install/) installed.
2. Navigate in your terminal to the directory you want this project to be installed to. Then clone the repository by running the following command.
``` bash
git clone https://github.com/Denomant/AI-Number-Guess.git
cd AI-Number-Guess
```
3. Create and activate a virtual environment.
``` bash
# Always replace python with python3 on Linux / macOS
python -m venv venv

# Windows PowerShell
.\venv\Scripts\activate
# Windows CMD
venv\Scripts\activate.bat
# Linux / macOS
source venv/bin/activate
```
4. Install prerequisites.
``` bash
pip install -r requirements.txt
```

### Logic Test Benchmarks
Run the following command:
``` bash
python TESTS.py
```

### Pygame Paint Application
Run the following command:
``` bash
python run.py
```

#### What is expected to work?
- Small 28x28 black-and-white painting screen.
- Brush size customization using the mouse wheel, and gradual shading as the size increases
- Erase screen button.
- Load button (Only loads 28x28 images; loading a trained AI is not implemented.).
	- Assuming [you decoded the MNIST dataset](#Download-MNIST), you can test it by clicking the load / import button, and typing `picture`
and `dataset_decoded\test\0\3` (`dataset_decoded/test/0/3` on Linux/macOS). It should display a 0 on the painting screen.

By default, AI learning is disabled and expected to return incorrect predictions. If you wish to experiment with the AI, uncomment the following lines
in `run.py` modify the amount of epochs (second parameter of the second function). Also you will need to download and decode the dataset, see
[Download MNIST](#Download-MNIST) section. Training is slow — expect 30 seconds multiplied by the number of epochs on start-up
allocated for learning.  

``` python 
113    train_data = initialize_MNISTDataPieces()
114    neural_network.train(train_data, epochs=1, batch_size=128, learning_rate=0.999, annealing_factor=0.999)
```
**P.S Even on 150 epochs and ~1 hour of learning, the AI still produces incorrect predictions.**

## Technical Detail
### Files and Classes
| Name | Description | File |
|------|-------------|------|
| `DataPiece(data)` | Dynamically provides input data for the `train` method in `NeuralNetwork`. | `ai.py` |
| `Layer(size, activation_function)` | Represents a single layer in `NeuralNetwork`. Responsible for storing and activating neurons. | `ai.py` |
| `WeightMatrix(layer1, layer2)` | Connects two layers via weighted connections and controls their communication. | `ai.py` |
| `ActivationFunction` | Abstract base class that enforces the presence of both a forward call and a derivative method. | `ai.py` |
| `NeuralNetwork(layer_sizes, activation_functions)` | Implements the forward pass, backpropagation, gradient descent, and training loop using all other classes in the module. | `ai.py` |
| `Linear` | Concrete implementation of `ActivationFunction`. | `ai.py` |
| `BinaryStep` | Concrete implementation of `ActivationFunction`. | `ai.py` |
| `Sigmoid` | Concrete implementation of `ActivationFunction`. | `ai.py` |
| `ReLU` | Concrete implementation of `ActivationFunction`. | `ai.py` |
| `Softmax` | Concrete implementation of `ActivationFunction`. Note: the derivative is approximated due to backpropagation design limitations. | `ai.py` |
| `Button(path, function, top_left_position)` | Simple UI button abstraction built on top of Pygame. Loads an image from disk, optionally scales it, tracks its screen rectangle, and calls the provided function when invoked. | `buttons.py` |
| `Popup(screen, left, top, width, height, *args, font_dir, title)` | Creates a modal on-screen popup with multiple text input fields. Automatically lays out input boxes, handles keyboard and mouse interaction, and returns a list of user-entered values on confirmation or `None` if cancelled. Used for basic configuration and file-selection prompts. | `buttons.py` |
| `initialize_buttons(screen, picture, ai)` | Initializes and positions all UI buttons relative to the screen size and returns them in a fixed order. Each button wraps a small handler function (exit, import, save, train, clear), some of which rely on `Popup` for user input. Note: saving, training, and loading neural network weights are not yet implemented. | `buttons.py` |
| `render(screen, picture, pixel_size)` | Renders a 28×28 grayscale image to the screen by drawing scaled rectangles for each pixel. Performs basic validation on the input array and draws a simple white frame around the image. | `render.py` |
| `draw_predictions(surface, predictions, start_x, start_y, bar_height, bar_width, spacing, font)` | Visualizes neural network output as horizontal confidence bars. Highlights the most confident prediction and optionally renders numeric labels and percentages using a provided font. | `render.py` |
| `paint_brush(picture, center_x, center_y, pixel_size, picture_size, radius)` | Simulates a soft circular brush on a 28×28 grayscale image. Increases pixel intensity based on distance from the brush center while ensuring pixels never become darker. | `render.py` |

Guiding design principle — To be able to create any possible Neural Network through an API similar to this:  
`ai.NeuralNetwork([784, 30, 20, 15, 10], [ai.Linear(), ai.Sigmoid(), ai.ReLU(), Custom(), ai.Softmax()])`

### Critical Flaws

- 🔴 Not learning MNIST dataset (exact cause unknown).
- 🔴 Weak Softmax derivative approximation due to brittle backpropagation design that wouldn't allow full Jacobian.
- 🔴 Lacking custom and modular cost functions.
- 🔴 No Xavier weight initialization (I learned about this technique after abandoning the project).
- 🔴 No GPU acceleration.

<!-- -->

- 🔴 Train, save, and load AI buttons are not implemented.
- 🟡 Brittle hand-made pop-up system.

<!-- -->

- 🟡 Inefficient memory > speed trade-off with dynamic image loading, especially in training with a lot of epochs.
- 🟡 ReLU could have behaved more efficiently if it used a small nonzero derivative on negatives.
- 🟡 Brittle validate_array design.
- 🟡 Many methods lack docstrings (project was abandoned before finishing).



## Download MNIST
Go to [The Kaggle website](https://www.kaggle.com/datasets/hojjatk/mnist-dataset?resource=download)
and download the .zip file. Then unzip it into the `./dataset_encoded/`
directory.  
1. Ensure that these 4 files are in the `./dataset_encoded/` directory, ready to be decoded:
   - `train-images.idx3-ubyte`
   - `train-labels.idx1-ubyte`
   - `t10k-images.idx3-ubyte`
   - `t10k-labels.idx1-ubyte`
2. [Assuming you have the requirements installed](#getting-started), run the following command to decode the dataset:
```bash
python dataset_decoder.py
```
