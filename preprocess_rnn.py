import pandas as pd
import preprocess
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
import nltk
import time
# from importlib import reload
# reload(preprocess)

# READ TWEETS
pd.set_option('display.max_columns', None)
train = pd.read_csv("./train.csv")[['target', 'comment_text']][:10000]
train['toxic'] = 1 * (train['target'] > 0.5)

# READ WORDS DATABASE
# https://nlp.stanford.edu/projects/glove/
w2v_dict = dict()
f = open("./glove.840B.300d.txt", "rt")
for i in tqdm(range(100000)):
    line = f.readline()
    els = line.split()
    token = els[0]
    value = tf.convert_to_tensor(list(map(float, els[1:])), dtype=tf.float32)
    w2v_dict[token] = value
f.close()

# TOKENIZE
vecs = []
for tox, text in tqdm(zip(train['toxic'], train['comment_text'])):
    vec = []
    for word in preprocess.proc(text, [1]):
        if word in w2v_dict:
            vec.append(w2v_dict[word])
    vecs.append((tox, vec))

## To execute the training process
  
optimizer = tf.keras.optimizers.Adam()

import rnn
from rnn import Encoder, Decoder, train_step
#reload(rnn)
BATCH_SIZE = 1
encoder = Encoder(300, 300, 100, BATCH_SIZE)
decoder = Decoder(300, 300, 100, BATCH_SIZE)

EPOCHS = 100

# Run this to train
def train_epoch():
    start = time.time()

    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0
    
    variables = encoder.trainable_variables + decoder.trainable_variables
    thing = tqdm(enumerate(vecs))
    for (batch_id, (tox, words)) in thing:
        # ignore empty tweets
        if not len(words):
            continue
        with tf.GradientTape() as tape:
            loss = train_step(words, encoder, decoder, enc_hidden)
            
            gradients = tape.gradient(loss, variables)   
            optimizer.apply_gradients(zip(gradients, variables))
            total_loss += loss
            if batch_id % 1 == 0:
                thing.set_description('Batch {} Loss {:.4f}'.format( batch_id, loss.numpy()))
            if batch_id % 100 == 0:
                encoder.save_weights("encoder.weights")

encoder.load_weights("./encoder.weights")

def tweet2embed(tweet):
    tweet = tf.stack(tweet)
    tweet = tf.expand_dims(tweet, axis=1)
    tweet = tf.expand_dims(tweet, axis=3)
    hid = encoder.initialize_hidden_state()
    for word in tweet:
        _, hid = encoder(word, hid)
    return hid

