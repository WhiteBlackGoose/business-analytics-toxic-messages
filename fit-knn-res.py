from preprocess import proc
import freq
from freq import read_kw, word_list2freq_dict
from importlib import reload
import pandas as pd
import numpy as np
from sklearn.metrics import fbeta_score
reload(freq)


# read keywords
kw = read_kw()

# load input
train = pd.read_csv("./train.csv")[['target', 'comment_text']][:200000]
test = pd.read_csv("./test.csv")[['comment_text']]
train['toxic'] = 1 * (train['target'] > 0.5)


# transform input comments to vectors
def comment_text_to_vec(comment_text):
    p = proc(comment_text, [1, 2])
    return word_list2freq_dict(kw, p)


train["comment"] = train['comment_text'].apply(comment_text_to_vec)


from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

# split into train and test samples
X_train, X_test, y_train, y_test = train_test_split(train["comment"], train["toxic"])

X_train = np.stack(X_train)
X_test = np.stack(X_test)

# scale data
scaler = StandardScaler()

Xs_train = scaler.fit_transform(X_train)
Xs_test = scaler.transform(X_test)

m = KNeighborsClassifier(n_neighbors=3, weights='uniform')
m.fit(Xs_train, y_train)
pred = m.predict(Xs_test)
pred_train = m.predict(Xs_train)

print(f1_score(y_test, pred))
print(f1_score(pred_train, y_train))
print(m.score(Xs_train, y_train))
print(m.score(Xs_test, y_test))