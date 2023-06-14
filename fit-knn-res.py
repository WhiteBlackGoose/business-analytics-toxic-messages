from preprocess import proc
import freq
from freq import read_kw, word_list2freq_dict
from importlib import reload
import pandas as pd
import numpy as np
from sklearn.metrics import fbeta_score
reload(freq)


def cross_validation(model, X, y, scorer, cv=5):
    return cross_validate(estimator=model,
                          X=X,
                          y=y,
                          cv=cv,
                          scoring=scorer,
                          return_train_score=True)


def scor(y, y_pred):
    t_pt = [i * j for i in y for j in y_pred]
    f_pf = [(1 - i) * (1 - j) for i in y for j in y_pred]
    return sum(t_pt) * sum(f_pf) / (sum(y_pred) * (len(y_pred) - sum(y_pred)))


kw = read_kw()

train = pd.read_csv("./train.csv")[['target', 'comment_text']][:200000]
test = pd.read_csv("./test.csv")[['comment_text']]
train['toxic'] = 1 * (train['target'] > 0.5)


def comment_text_to_vec(comment_text):
    p = proc(comment_text, [1, 2])
    return word_list2freq_dict(kw, p)


train["comment"] = train['comment_text'].apply(comment_text_to_vec)


from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(train["comment"], train["toxic"])

X_train = np.stack(X_train)
X_test = np.stack(X_test)
scaler = StandardScaler()

Xs_train = scaler.fit_transform(X_train)
Xs_test = scaler.transform(X_test)

m = KNeighborsClassifier(n_neighbors=5, weights='distance')
m.fit(Xs_train, y_train)
pred = m.predict(Xs_test)
pred_train = m.predict(Xs_train)

print(f1_score(y_test, pred))
print(f1_score(pred_train, y_train))
print(m.score(Xs_train, y_train))
print(m.score(Xs_test, y_test))