from rnn import Encoder
from preprocess import proc
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import f1_score

encoder = Encoder(300, 300, 100, 1)
encoder.load_weights("encoder.weights")

# Collect all words into dictionary
# to quickly fetch a vector given word
w2v_dict = dict()
f = open("./glove.840B.300d.txt", "rt", encoding='utf-8')
for i in tqdm(range(100000)):
    line = f.readline()
    els = line.split()
    token = els[0]
    try:
        value = tf.convert_to_tensor(list(map(float, els[1:])), dtype=tf.float32)
    except:
        break
    w2v_dict[token] = value
f.close()

# Runs encoder on a commext text
def comment_text_to_vec(comment_text):
    p = proc(comment_text, [1, 2])
    tweet = []
    for word in p:
        if word in w2v_dict:
            tweet.append(w2v_dict[word])
    if len(tweet) == 0:
        return None
    tweet = tf.stack(tweet)
    tweet = tf.expand_dims(tweet, axis=1)
    tweet = tf.expand_dims(tweet, axis=3)
    enc_hidden = encoder.initialize_hidden_state()
    for word in tweet:
        out_t, enc_hidden = encoder(word, enc_hidden)
    return enc_hidden[0].numpy()


train = pd.read_csv("./train.csv")[['target', 'comment_text']][:10000]
train['toxic'] = 1 * (train['target'] > 0.5)

print("Training")
X, y = [], []
for Xrow, yrow in tqdm(list(zip(train['comment_text'], train["toxic"]))[:50]):
    v = comment_text_to_vec(Xrow)
    if v is not None:
        X.append(v)
        y.append(yrow)

import numpy as np
X = np.stack(X)
y = np.array(y)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)
m = AdaBoostClassifier()
m.fit(X_train, y_train)
print("Train accuracy:", m.score(X_train, y_train))
print("Test accuracy:", m.score(X_test, y_test))
print("Train f1:", f1_score(y_train, m.predict(X_train)))
print("Test f1:", f1_score(y_test, m.predict(X_test)))
