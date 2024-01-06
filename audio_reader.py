import tensorflow as tf
from os.path import join as pjoin
import numpy as np
import glob
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt

class WavFileReader:
    def __init__(self,wav_length):
        self.data = []
        self.samplerate = None
        self.wav_length=wav_length

    def read_wav_files(self, wav_filenames, reader_function):
        for wav in wav_filenames:
            result, samplerate = reader_function(wav)
            self.data.append(result)
            if self.samplerate is None:
                self.samplerate = samplerate
        return self.data

    def wavIR_reader(self, wav_filename):
        samplerate, y = wavfile.read(wav_filename)
        idx = np.argmax(y)
        return y[idx:idx+self.wav_length], samplerate


    def plot_frequency_response(self, data, subplot_index):
        w, h = signal.freqz(b=data, a=1)
        x = w * self.samplerate * 1.0 / (2 * np.pi)
        y = 20 * np.log10(abs(h))
        plt.subplot(self.rows, self.cols, subplot_index)
        plt.semilogx(x, y)
        plt.ylabel('Amplitude [dB]')
        plt.xlabel('Frequency [Hz]')
        plt.title(f'Frequency response - Signal {subplot_index}')
        plt.grid(which='both', linestyle='-', color='grey')
        plt.xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000],
                   ["20", "50", "100", "200", "500", "1K", "2K", "5K", "10K", "20K"])


    def plot_waveform(self, data):
        length = len(data) / self.samplerate
        time = np.linspace(0., length, len(data))
        plt.plot(time, data, label="one channel")
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.show()


    def plot_multiple_frequency_responses(self, dataset, rows, cols):
        self.rows = rows
        self.cols = cols
        plt.figure(figsize=(12, 6))

        for batch_idx, signal_batch in enumerate(dataset, start=1):
            for signal_idx, signal_data in enumerate(signal_batch, start=1):
                signal_data_np = np.array(signal_data)
                idx = (batch_idx - 1) * len(signal_batch) + signal_idx
                self.plot_frequency_response(signal_data_np, idx)
            print(batch_idx)

        plt.tight_layout()
        plt.show()

def main():
    # Example usage
    data_dir = 'Marshall1960A_1105'
    wav_length=512
    batch_size=1

    wav_reader = WavFileReader(wav_length)
    wav_files = glob.glob(pjoin(data_dir, '**', '*wav'), recursive=True)
    wav_reader.read_wav_files(wav_files, wav_reader.wavIR_reader)
    print(np.shape(wav_reader.data))

    # train_dataset = wav_reader.create_dataset().batch(batch_size)
    train_dataset=tf.data.Dataset.from_tensor_slices(wav_reader.data).batch(batch_size)
    wav_reader.plot_multiple_frequency_responses(train_dataset,rows=6, cols=6)  # Specify the number of rows and columns

    print("Shape of the train_dataset element:", train_dataset.element_spec)
if __name__ == "__main__":
    main()

# for i in train_dataset:
#     wav_reader.plot_waveform(i)
