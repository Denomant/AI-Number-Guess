# AI-Number-Guess
## Table of Contents
- [Getting started](#Getting-started)

## Getting started
### Downloading the dataset
Go to [The Kaggle website](https://www.kaggle.com/datasets/hojjatk/mnist-dataset?resource=download)
and follow their instructions to download the dataset into the
`./dataset_decoded/` directory, or download the .zip file
and unzip in into the `./dataset_decoded/` directory, then my follow
instruction below to decode it.

### Decoding the dataset
1. Ensure that there are 4 files in the `./dataset_decoded/` directory:
   - `train-images.idx3-ubyte`
   - `train-labels.idx1-ubyte`
   - `t10k-images.idx3-ubyte`
   - `t10k-labels.idx1-ubyte`
2. Download the requirements by running the following command:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the following command to decode the dataset:
   ```bash
   python dataset_decoder.py
   ```