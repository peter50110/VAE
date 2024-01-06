import argparse
import tensorflow as tf
import numpy as np
import glob
import matplotlib.pyplot as plt
from audio_reader import WavFileReader
from scipy.io.wavfile import write
from VAEmodel import build_encoder_decoder,sampling,ELBO_loss

# Function to load the model and perform evaluation
def evaluate_model(encoder_weights_path,decoder_weights_path ,data_dir, seq_size ,num_hidden_units, latent_dim,batch_size,output_path):
    # Build the encoder and decoder networks
    encoder_net, decoder_net = build_encoder_decoder(seq_size, num_hidden_units, latent_dim)

    # Load the saved weights
    encoder_net.load_weights(encoder_weights_path)
    decoder_net.load_weights(decoder_weights_path)
    encoder_net.summary()
    # Create WavFileReader instance
    wav_reader = WavFileReader(seq_size)
    wav_files = glob.glob(f'{data_dir}/**/*.wav', recursive=True)
    wav_reader.read_wav_files(wav_files, wav_reader.wavIR_reader)

    # Create TensorFlow dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(wav_reader.data).batch(batch_size)

    # Assuming XtestData is your test data
    for batch in train_dataset.take(1):  # Take one batch for visualization
        XtestData = batch

    # Assuming XtestData is your test data
    z_sampled, z_mean, z_logvar = sampling(encoder_net, XtestData)
    x_pred = decoder_net.predict(z_sampled)

    # Save the reconstructed signal as a WAV file
    x_pred=x_pred[0]
    scaled_x_pred = x_pred / np.max(np.abs(x_pred))
    write(output_path + 'output_reconstructed.wav', wav_reader.samplerate, scaled_x_pred.astype(np.float32))

    # Plot the original signal
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(XtestData[0].numpy())  # Assuming XtestData is a batch of signals
    plt.title('Original Signal')

    # Plot the reconstructed signal
    plt.subplot(1, 2, 2)
    plt.plot(x_pred)
    plt.title('Reconstructed Signal')

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate an autoencoder for audio signals.')
    parser.add_argument('--encoder_weights_path', type=str, help='Path to the saved weights file')
    parser.add_argument('--decoder_weights_path', type=str, help='Path to the saved weights file')
    parser.add_argument('--data_dir', type=str, help='Directory containing audio files')
    parser.add_argument('--seq_size', type=int, default=512, help='Sequence size for audio signals')
    parser.add_argument('--num_hidden_units', type=int, default=128, help='Number of hidden units in the networks')
    parser.add_argument('--latent_dim', type=int, default=40, help='Latent dimension of the autoencoder')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
    parser.add_argument('--output_path', type=str, default='output/', help='Output path for saving results')
    args = parser.parse_args()

    evaluate_model(args.encoder_weights_path,args.decoder_weights_path, args.data_dir, args.seq_size, args.num_hidden_units, args.latent_dim,
                   args.batch_size, args.output_path)


#python load_model.py --encoder_weights_path output/best_encoder_weights_.h5 --decoder_weights_path output/best_decoder_weights_.h5 --data_dir Marshall1960A_1105/ --seq_size 16000 --num_hidden_units 128 --latent_dim 40
