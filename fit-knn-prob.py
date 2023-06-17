from preprocess import proc
import freq
from freq import read_kw, word_list2freq_dict
from importlib import reload
import pandas as pd
import numpy as np
from sklearn.metrics import fbeta_score
reload(freq)

# function for performing cross-validation
def cross_validation(model, X, y, scorer, cv=5):
    return cross_validate(estimator=model,
                          X=X,
                          y=y,
                          cv=cv,
                          scoring=scorer,
                          return_train_score=True)


# get keywords
kw = read_kw()

# load train data
train = pd.read_csv("./train.csv")[['target', 'comment_text']][:20000]
train['toxic'] = 1 * (train['target'] > 0.5)

# transform input to vector
def comment_text_to_vec(comment_text):
    p = proc(comment_text, [1, 2])
    return word_list2freq_dict(kw, p)


train["comment"] = train['comment_text'].apply(comment_text_to_vec)


from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neighbors import KNeighborsClassifier


X = np.stack(train["comment"])
y = train["toxic"]

# gridsearch + crossvalidation for choosing optimal hyperparameters
best = 0
k_best = 0
weight_best = 0
for k in range(1, 15):
    for weights in ('uniform', 'distance'):
        m = KNeighborsClassifier(n_neighbors=k, weights=weights)
        res = cross_validation(m, X, y, ['f1'], 5)
        if res['test_f1'].mean() > best:
            best = res['test_f1'].mean()
            k_best = k
            weight_best = weights

print(best, k_best, weight_best)