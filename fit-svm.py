from preprocess import proc
import freq
import pandas as pd
from freq import read_kw, word_list2freq_dict
from importlib import reload
from sklearn.metrics import f1_score
import pandas as pd

kw = read_kw()

balanced = False

if balanced:
    train = pd.read_csv("./train.csv")[['target', 'comment_text']]
    print("Dataset read")
    train['toxic'] = 1 * (train['target'] > 0.5)

    train_toxic = train[train['toxic'] == 1]
    train_not_toxic = train[train['toxic'] == 0].sample(len(train_toxic))
    train = pd.concat([train_toxic, train_not_toxic]).sample(len(train_toxic) * 2)
    print("Manipulations performed")
else:
    train = pd.read_csv("./train.csv")[['target', 'comment_text']][:100000]
    print("Dataset read")
    train['toxic'] = 1 * (train['target'] > 0.5)

def comment_text_to_vec(comment_text):
    p = proc(comment_text, [1, 2])
    return word_list2freq_dict(kw, p)

print("Stage 1/3")

train["comment"] = train['comment_text'].apply(comment_text_to_vec)

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

print("Stage 2/3")

X_train, X_test, y_train, y_test = train_test_split(train["comment"], train["toxic"])

import numpy as np
X_train = np.stack(X_train)
X_test = np.stack(X_test)

print("Stage 3/3")

m = SVC(verbose=1)
m.fit(X_train, y_train)
print("Train accuracy:", m.score(X_train, y_train))
print("Test accuracy:", m.score(X_test, y_test))
print("Train f1:", f1_score(y_train, m.predict(X_train)))
print("Test f1:", f1_score(y_test, m.predict(X_test)))
