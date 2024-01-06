import tensorflow as tf
from os.path import join as pjoin
import numpy as np
from audio_reader import *
from scipy import signal
import matplotlib.pyplot as plt

data_dir = 'Marshall1960A_1105'
wav_reader = WavFileReader()
wav_files = glob.glob(pjoin(data_dir, '**', '*wav'), recursive=True)
wav_reader.read_wav_files(wav_files, wav_reader.wavIR_reader)
print(pjoin(data_dir, '**', '*wav'))
print(np.shape(wav_reader.data))
train_dataset = tf.data.Dataset.from_tensor_slices(wav_reader.data)


for i in train_dataset:
  print(np.shape(i))
  w, h = signal.freqz(b=i, a=1)
  x = w * 96000 * 1.0 / (2 * np.pi)
  y = 20 * np.log10(abs(h))
  plt.semilogx(x, y)
  plt.ylabel('Amplitude [dB]')
  plt.xlabel('Frequency [Hz]')
  plt.title('Frequency response')
  plt.grid(which='both', linestyle='-', color='grey')
  plt.xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000], ["20", "50", "100", "200", "500", "1K", "2K", "5K", "10K", "20K"])
  plt.show()

for i in train_dataset:
  print(np.shape(i))
  length = i.shape[0] / wav_reader.samplerate
  time = np.linspace(0., length, i.shape[0])
  plt.plot(time, i, label="one channel")
  plt.legend()
  plt.xlabel("Time [s]")
  plt.ylabel("Amplitude")
  plt.show()
