from rnn import Encoder
from preprocess import proc
import tensorflow as tf
from tqdm import tqdm
import pandas as pd

encoder = Encoder(300, 300, 100, 1)
encoder.load_weights("encoder.weights")

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


def cross_validation(model, X, y, scorer, cv=5):
    return cross_validate(estimator=model,
                          X=X,
                          y=y,
                          cv=cv,
                          scoring=scorer,
                          return_train_score=True)


train = pd.read_csv("./train.csv")[['target', 'comment_text']][:10000]
train['toxic'] = 1 * (train['target'] > 0.5)

print("Training")
X, y = [], []
for Xrow, yrow in tqdm(list(zip(train['comment_text'], train["toxic"]))[:5000]):
    v = comment_text_to_vec(Xrow)
    if v is not None:
        X.append(v)
        y.append(yrow)

import numpy as np
X = np.stack(X)
y = np.array(y)



from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

best = 0
pen_best = 0
inter_best = 0
for penalty in ('l1', 'l2', 'elasticnet', 'none'):
    for intercept in (True, False):
        m = LogisticRegression(penalty = penalty, fit_intercept = intercept)
        res = cross_validation(m, Xs, y, ['f1'], 5)
        if res['test_f1'].mean() > best:
            best = res['test_f1'].mean()
            pen_best = penalty
            inter_best = intercept

print(best, pen_best, inter_best)
