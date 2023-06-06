import pandas as pd
from preprocess import proc
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

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

model = keras.Sequential()
# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.
model.add(layers.Embedding(input_dim=1000, output_dim=64))

# Add a LSTM layer with 128 internal units.
model.add(layers.LSTM(128))

# Add a Dense layer with 10 units.
model.add(layers.Dense(10))

model.summary()
