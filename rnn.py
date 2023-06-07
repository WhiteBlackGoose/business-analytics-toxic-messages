import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
# https://towardsdatascience.com/sequence-to-sequence-models-from-rnn-to-transformers-e24097069639

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        '''
        vocab_size: number of unique words
        embedding_dim: dimension of your embedding output
        enc_units: how many units of RNN cell
        batch_sz: batch of data passed to the training in each epoch
        '''
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        _, state = self.gru(x, initial_state = hidden)
        return state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)


    def call(self, hidden):
        # as we have specified return_sequences=True in encoder gru
        # enc_output shape == (batch_size, max_length, hidden_size)

        output, state = self.gru(hidden)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)
    
        return x, state


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


@tf.function
def train_step(tweet, encoder, decoder, enc_hidden):
    loss = 0
    with tf.GradientTape() as tape:
        dec_hidden = enc_hidden
        for word in tweet:
            enc_hidden = encoder(tf.expand_dims(tf.expand_dims(word, axis=1), axis=0), enc_hidden)
            dec_output, dec_hidden = decoder(dec_hidden)
            loss += loss_function(enc_output, dec_output)
    return loss
