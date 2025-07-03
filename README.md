# AI-Number-Guess
## Table of Contents
- [Getting started](#Getting-started)

## Getting started
### Downloading the MNIST dataset
Go to [The Kaggle website](https://www.kaggle.com/datasets/hojjatk/mnist-dataset?resource=download)
and download the .zip file, then unzip in into the `./dataset_encoded/`
directory, and then follow my instruction below to decode it.

### Decoding the dataset
1. Ensure that these 4 files are in the `./dataset_decoded/` directory:
   - `train-images.idx3-ubyte`
   - `train-labels.idx1-ubyte`
   - `t10k-images.idx3-ubyte`
   - `t10k-labels.idx1-ubyte`
2. Download the requirements by running the following command (if haven't done it yet):
   ```bash
   pip install -r requirements.txt
   ```
3. Run the following command to decode the dataset:
   ```bash
   python dataset_decoder.py
   ```