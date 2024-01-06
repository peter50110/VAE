---

# Autoencoder for Audio Signals

## Overview

This repository contains a Python script for training an autoencoder on audio signals. The autoencoder is composed of an encoder and a decoder neural network, and it is trained to reconstruct audio signals in a latent space.

## Files

- `audio_reader.py`: Python script containing a class `WavFileReader` for reading and processing audio signals.
- `model.py`: Python script containing functions to build the encoder and decoder neural networks.
- `train_autoencoder.py`: Main Python script for training the autoencoder.

## Dependencies

- TensorFlow: [Install TensorFlow](https://www.tensorflow.org/install).
- NumPy: Install using `pip install numpy`.
- SciPy: Install using `pip install scipy`.
- Matplotlib: Install using `pip install matplotlib`.

## Usage

### 1. Clone the Repository:

```bash
git clone https://github.com/your_username/autoencoder-audio.git
cd autoencoder-audio
```

### 2. Install Dependencies:

```bash
pip install -r requirements.txt
```

### 3. Run the Training Script:

```bash
python train_autoencoder.py
```

- Adjust hyperparameters in `train_autoencoder.py` as needed (e.g., `num_epochs`, `batch_size`, etc.).

### 4. Evaluation:

After training, the script will visualize the training loss, learning rate, and evaluate the reconstructed signals. The best model will be saved with the lowest loss.

### 5. Reconstructed Signal Output:

The reconstructed signals will be saved in the file `output_reconstructed.wav`.

## Customization

- Modify hyperparameters, such as `seq_size`, `num_hidden_units`, `latent_dim`, `batch_size`, and `num_epochs` in `train_autoencoder.py` based on your requirements.
- Customize the autoencoder architecture by modifying functions in `model.py`.

## Details

### `audio_reader.py`

The `WavFileReader` class reads and processes audio signals using the `scipy` library. It provides methods to create a TensorFlow dataset from a list of WAV files.

### `model.py`

The `build_encoder_decoder` function in this file creates the encoder and decoder neural networks. The autoencoder architecture can be customized by adjusting the `seq_size`, `num_hidden_units`, and `latent_dim` parameters.

### `train_autoencoder.py`

This script is the main entry point for training the autoencoder. It uses the encoder and decoder networks, defines a custom training loop, and saves the best model based on the lowest training loss.

Example exe .py: python train_autoencoder.py --data_dir Marshall1960A_1105/ --seq_size 512 --num_hidden_units 128 --latent_dim 40 --batch_size 4 --num_epochs 2000 --output_path /

## Author

- Richard Tsai
- Contact: yuttsai@fcu.edu.tw

---
