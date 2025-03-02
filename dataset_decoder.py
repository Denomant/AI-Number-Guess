import os
import os.path
import numpy as np
import cv2

"""
This is the script that decodes the MNIST dataset from the .ubyte files to .png images.
"""

def save_mnist_images(image_file, label_file, output_dir):
    with open(image_file, 'rb') as img_f, open(label_file, 'rb') as lbl_f:
        img_f.read(16)  # Skip the header
        lbl_f.read(8)   # Skip the header
        images = np.frombuffer(img_f.read(), dtype=np.uint8).reshape(-1, 28, 28)
        labels = np.frombuffer(lbl_f.read(), dtype=np.uint8)

        for i, (image, label) in enumerate(zip(images, labels)):
            label_dir = os.path.join(output_dir, str(label))
            os.makedirs(label_dir, exist_ok=True)
            image_path = os.path.join(label_dir, f'{i}.png')
            cv2.imwrite(image_path, image)

# Paths to the extracted .ubyte files

train_images = os.path.join('dataset_encoded', 'train-images.idx3-ubyte')
train_labels = os.path.join( 'dataset_encoded', 'train-labels.idx1-ubyte')
test_images = os.path.join('dataset_encoded', 't10k-images.idx3-ubyte')
test_labels = os.path.join('dataset_encoded', 't10k-labels.idx1-ubyte')

# Output directories
train_output = os.path.join('dataset_decoded', 'train')
test_output = os.path.join('dataset_decoded', 'test')

save_mnist_images(train_images, train_labels, train_output)
save_mnist_images(test_images, test_labels, test_output)
