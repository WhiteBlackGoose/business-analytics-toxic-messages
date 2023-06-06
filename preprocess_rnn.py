import pandas as pd
import preprocess
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
import nltk
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
    value = np.array(list(map(float, els[1:])))
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
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

from rnn import Encoder, Decoder
BATCH_SIZE = 10
encoder = Encoder(300, 300, 100, BATCH_SIZE)
decoder = Decoder(300, 300, 100, BATCH_SIZE)

EPOCHS = 100

for epoch in range(EPOCHS):
    start = time.time()

    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch_id, (tox, words)) in enumerate(vecs):
        batch_loss = train_step(words, encoder, decoder, enc_hidden)
        total_loss += batch_loss

        if batch_id % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                   batch_id,
                                                   batch_loss.numpy()))
    # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)

    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
