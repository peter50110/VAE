import argparse
from audio_reader import WavFileReader
from VAEmodel import build_encoder_decoder,sampling,ELBO_loss
import tensorflow as tf
import numpy as np
import glob
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from datetime import datetime

# Function to train the autoencoder
def train_autoencoder(data_dir, seq_size, num_hidden_units, latent_dim, batch_size, num_epochs, output_path):
    # Build the encoder and decoder networks
    encoder_net, decoder_net = build_encoder_decoder(seq_size, num_hidden_units, latent_dim)

    # Create optimizer with exponential decay learning rate
    initial_learning_rate = 1e-3
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=4000, decay_rate=0.9, staircase=True
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # Create WavFileReader instance
    wav_reader = WavFileReader(seq_size)
    wav_files = glob.glob(f'{data_dir}/**/*.wav', recursive=True)
    wav_reader.read_wav_files(wav_files, wav_reader.wavIR_reader)
    print(np.shape(wav_reader.data))

    # Create TensorFlow dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(wav_reader.data).batch(batch_size)

    # Lists to store loss values and learning rates
    losses = []
    learning_rates = []
    # Initialize variables for tracking the lowest loss and corresponding epoch
    lowest_loss = float('inf')
    best_epoch = -1

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in train_dataset:
            loss = train_step(encoder_net, decoder_net, optimizer, batch)
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

        # Save the model with a timestamp in the filename
        # with open(f'{output_path}model_config_{timestamp}.json', 'w') as f:
        #     f.write(encoder_net.to_json())
        # encoder_net.save_weights(f'{output_path}best_encoder_weights_{timestamp}.h5')
        # decoder_net.save_weights(f'{output_path}best_decoder_weights_{timestamp}.h5')

        encoder_net.save_weights(f'{output_path}best_encoder_weights_.h5')
        decoder_net.save_weights(f'{output_path}best_decoder_weights_.h5')

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

    # Evaluation
    # After training, you can use the encoder and decoder networks as needed
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

    # Save the reconstructed signal as a WAV file
    x_pred=x_pred[0]
    scaled_x_pred = x_pred / np.max(np.abs(x_pred))
    write(output_path + 'output_reconstructed.wav', wav_reader.samplerate, scaled_x_pred.astype(np.float32))



# Function to train one step of the autoencoder
@tf.function
def train_step(encoder_net, decoder_net, optimizer, x):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an autoencoder for audio signals.')
    parser.add_argument('--data_dir', type=str, help='Directory containing audio files')
    parser.add_argument('--seq_size', type=int, default=512, help='Sequence size for audio signals')
    parser.add_argument('--num_hidden_units', type=int, default=128, help='Number of hidden units in the networks')
    parser.add_argument('--latent_dim', type=int, default=40, help='Latent dimension of the autoencoder')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=2000, help='Number of training epochs')
    parser.add_argument('--output_path', type=str, default='output/', help='Output path for saving model and results')
    args = parser.parse_args()

    train_autoencoder(args.data_dir, args.seq_size, args.num_hidden_units, args.latent_dim,
                       args.batch_size, args.num_epochs, args.output_path)

#python train_autoencoder.py --data_dir Marshall1960A_1105/ --seq_size 512 --num_hidden_units 128 --latent_dim 40 --batch_size 4 --num_epochs 10