from preprocess import proc
import freq
import pandas as pd
from freq import read_kw, word_list2freq_dict
from importlib import reload
reload(freq)

kw = read_kw()

train = pd.read_csv("./train.csv")[['target', 'comment_text']][:10000]
train['toxic'] = 1 * (train['target'] > 0.5)

def comment_text_to_vec(comment_text):
    p = proc(comment_text, [1, 2])
    return word_list2freq_dict(kw, p)

train["comment"] = train['comment_text'].apply(comment_text_to_vec)

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train["comment"], train["toxic"])

import numpy as np
X_train = np.stack(X_train)
X_test = np.stack(X_test)

m = SVC()
m.fit(X_train, y_train)
m.score(X_train, y_train)
m.score(X_test, y_test)
