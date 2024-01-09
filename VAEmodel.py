from keras.models import Sequential, Model
from keras.layers import Input, LSTM, Dense, Bidirectional, Dropout, Activation, BatchNormalization
import tensorflow as tf
from keras.optimizers import Adam

def sampling(encoder_net, x):
    compressed = encoder_net(x)
    d = compressed.shape[1] // 2
    z_mean = compressed[:, :d]
    z_logvar = compressed[:, d:]

    sz = tf.shape(z_mean)
    epsilon = tf.random.normal(sz)
    sigma = tf.exp(0.5 * z_logvar)
    z = epsilon * sigma + z_mean
    z_sampled = z

    return z_sampled, z_mean, z_logvar

def ELBO_loss(x, x_pred, z_mean, z_logvar):
    batch_size = tf.shape(x)[0]

    x_pred = tf.cast(x_pred, tf.float32)
    x = tf.cast(x, tf.float32)

    squares = 0.5 * tf.reduce_sum(tf.square(x_pred - x)) / tf.cast(batch_size, tf.float32)
    reconstruction_loss = tf.reduce_sum(squares)

    kl = -0.5 * tf.reduce_sum(1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar))

   # beta = 1 / tf.cast(batch_size, tf.float32) #test 1
    beta = 50 #test
    beta *= tf.reduce_mean(kl)
    
    elbo = reconstruction_loss + beta

    return elbo, reconstruction_loss, beta

def build_encoder_decoder(seq_size, num_hidden_units, latent_dim, dropout_rate=0.5):
    # Encoder
    encoder_model = Sequential(name='encoder')
    encoder_model.add(Input(shape=(seq_size,1)))
    encoder_model.add(Activation('tanh'))
    encoder_model.add(Dense(num_hidden_units, name='fc1'))
    encoder_model.add(BatchNormalization())
    encoder_model.add(Activation('tanh'))
    encoder_model.add(Dropout(dropout_rate))
    
    # Add LSTM layer to the encoder
    encoder_model.add(LSTM(num_hidden_units, return_sequences=True))

    # Decoder
    decoder_model = Sequential(name='decoder')
    decoder_model.add(Input(shape=(latent_dim,)))
    decoder_model.add(Activation('tanh'))
    decoder_model.add(Dense(num_hidden_units, name='fc_1'))
    decoder_model.add(BatchNormalization())
    decoder_model.add(Activation('tanh'))
    decoder_model.add(Dropout(dropout_rate))

    # Add LSTM layer to the decoder
    decoder_model.add(LSTM(num_hidden_units, return_sequences=True))

    return encoder_model, decoder_model

#Example test:
# Specify the input sizes
seq_size = 512 # Set the sequence size
num_hidden_units =64  # Set the number of hidden units
latent_dim = 40 # Set the latent dimension


# Build the encoder and decoder networks
encoder_net, decoder_net = build_encoder_decoder(seq_size, num_hidden_units, latent_dim)

# Print model summaries
encoder_net.summary()
decoder_net.summary()

