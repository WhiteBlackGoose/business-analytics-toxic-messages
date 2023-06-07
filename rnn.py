import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
# https://towardsdatascience.com/sequence-to-sequence-models-from-rnn-to-transformers-e24097069639
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

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
        y, hidden_out = self.gru(x, initial_state = hidden)
        return y, hidden_out

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


    def call(self, x, hidden):
        # as we have specified return_sequences=True in encoder gru
        # enc_output shape == (batch_size, max_length, hidden_size)

        output, state = self.gru(x, initial_state = hidden)

        # output shape == (batch_size * 1, hidden_size)
        #output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        #x = self.fc(output)
    
        return output, state


#def loss_function(real, pred):
#    mse = tf.keras.losses.MeanSquaredError()
#    return mse(real, pred)
loss_function = tf.keras.losses.MeanSquaredError()


@tf.function
def train_step(tweet, encoder, decoder, enc_hidden):
    tweet = tf.stack(tweet)
    tweet = tf.expand_dims(tweet, axis=1)
    tweet = tf.expand_dims(tweet, axis=3)
    for word in tweet:
        out_t, enc_hidden = encoder(word, enc_hidden)

    loss = 0.0
    dec_input = tf.ones(word.shape)
    for word in tweet:
        dec_input, enc_hidden = decoder(dec_input, enc_hidden)
        dec_input = tf.reshape(dec_input, (100, 300))[0]
        dec_input = tf.reshape(dec_input, word.shape)
        
        loss += loss_function(dec_input, word)
    return loss
