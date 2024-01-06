import tensorflow as tf
from os.path import join as pjoin
import numpy as np
import glob
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
from audio_reader import *
from VAEmodel import *
import matplotlib.pyplot as plt

# Hyperparameters
seq_size = 512  # Adjust based on your sequence size
num_hidden_units = 128  # Adjust based on your requirements
latent_dim = 40  # Adjust based on your requirements
batch_size = 1  # Adjust based on your preferences
num_epochs = 2000  # Adjust based on your preferences

# Build the encoder and decoder networks
encoder_net, decoder_net = build_encoder_decoder(seq_size, num_hidden_units, latent_dim)

# Create optimizer
initial_learning_rate = 1e-3
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=10000, decay_rate=0.9, staircase=True
)

optimizer = Adam(learning_rate=lr_schedule)

# Custom train loop
@tf.function
def train_step(x):
    with tf.GradientTape() as tape:
        # Forward pass
        z_sampled, z_mean, z_logvar = sampling(encoder_net, x)
        x_pred = decoder_net(z_sampled)
        
        # Compute loss
        loss = ELBO_loss(x, x_pred, z_mean, z_logvar)

    # Compute gradients and apply updates
    gradients = tape.gradient(loss, encoder_net.trainable_variables + decoder_net.trainable_variables)
    optimizer.apply_gradients(zip(gradients, encoder_net.trainable_variables + decoder_net.trainable_variables))

    return loss

# Assuming XtrainData is your training data
data_dir = 'Marshall1960A_1105'


wav_reader = WavFileReader(seq_size)
wav_files = glob.glob(pjoin(data_dir, '**', '*wav'), recursive=True)
wav_reader.read_wav_files(wav_files, wav_reader.wavIR_reader)
print(np.shape(wav_reader.data))


train_dataset=tf.data.Dataset.from_tensor_slices(wav_reader.data).batch(batch_size)

# Lists to store loss values and learning rates
losses = []
learning_rates = []
# Initialize variables for tracking the lowest loss and corresponding epoch
lowest_loss = float('inf')
best_epoch = -1

# Training loop
for epoch in range(num_epochs):
    total_loss = 0.0
    for batch in train_dataset:
        loss = train_step(batch)
        total_loss += loss

    average_loss = total_loss / len(train_dataset)
     # Access the current learning rate
    current_learning_rate = optimizer.learning_rate.numpy()

    # Append values to lists for plotting
    losses.append(average_loss.numpy())
    learning_rates.append(current_learning_rate)
    print(f'Epoch {epoch + 1}, Loss: {average_loss.numpy()}, Learning Rate: {current_learning_rate}')
    # Check if the current loss is the lowest encountered so far
    if average_loss < lowest_loss:
        lowest_loss = average_loss
        best_epoch = epoch

        # Save the model
        encoder_net.save_weights('best_encoder_weights.h5')
        decoder_net.save_weights('best_decoder_weights.h5')

print(f"Best model saved at epoch {best_epoch + 1} with the lowest loss: {lowest_loss.numpy()}")
# Plotting the loss values
plt.figure(figsize=(10, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Plot learning rate
plt.subplot(1, 2, 2)
plt.plot(learning_rates)
plt.title('Learning Rate')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')

plt.tight_layout()
plt.show()

#Evaluation!
# After training, you can use the encoder and decoder networks as needed
# Assuming train_dataset is your training dataset
for batch in train_dataset.take(1):  # Take one batch for visualization
    XtestData = batch

# Assuming XtestData is your test data
z_sampled, z_mean, z_logvar = sampling(encoder_net, XtestData)
x_pred = decoder_net.predict(z_sampled)

# Plot the original signal
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(XtestData[0].numpy())  # Assuming XtestData is a batch of signals
plt.title('Original Signal')

# Plot the reconstructed signal
plt.subplot(1, 2, 2)
plt.plot(x_pred[0])
plt.title('Reconstructed Signal')

plt.show()

from scipy.io.wavfile import write

# Assuming x_pred is a NumPy array containing the reconstructed signal
# Make sure to scale x_pred to the appropriate range, e.g., [-1, 1]
scaled_x_pred = x_pred / np.max(np.abs(x_pred))

# Define the output file path
output_file_path = 'output_reconstructed.wav'

# Set the sampling rate
sampling_rate = wav_reader.samplerate

# Save the waveform as a single-channel WAV file
write(output_file_path, sampling_rate, scaled_x_pred.astype(np.float32))
