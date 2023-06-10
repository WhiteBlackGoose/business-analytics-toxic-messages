import pandas as pd
from preprocess import proc
pd.set_option('display.max_columns', None)

train = pd.read_csv("./train.csv")[['target', 'comment_text']][:10000]
train['toxic'] = 1 * (train['target'] > 0.5)
# test = pd.read_csv("./test.csv")[['comment_text']]

from tqdm import tqdm

all_words = set()
for text in tqdm(train['comment_text']):
    all_words = all_words.union(proc(text, [1, 2]))

# P(T | C) != P(!T | C)
#
#       T           !T
#
# C    100         1000
#
#
# !C    1            10
#


s = dict()
for text, tox in tqdm(zip(train['comment_text'], train['toxic'])):
    words = proc(text, [1, 2])
    for word in words:
        if word not in s:
            s[word] = [0, 0]
        s[word][tox] += 1

NT = train.shape[0] - sum(train['toxic'])
T = sum(train['toxic'])

def cond_prob_tox(word):
    #P(M is toxic | word in M) = P(word in M | M is toxic) * P(M is toxic) / P(word in M)
    p_word_in_M = sum(s[word])/(T + NT)
    p_M_toxic = T/(NT + T)
    p_in_M_given_toxic = s[word][1]/T
    return p_in_M_given_toxic * p_M_toxic / p_word_in_M


sig = dict()
sig_list = []
for word in s:
    nt, t = s[word]
    if ' ' in word:
        if nt + t < 10:
            continue
    else:
        if nt + t < 25:
            continue
    value = (nt / NT - t / T) / (nt + t) * (NT + T)
    sig[word] = value
    sig_list.append((word, value, nt, t))



sig_list_sorted = sorted(sig_list, key=lambda x: -abs(x[1]))
sig_list_sorted_pos = sorted(list(filter(lambda x: x[1] > 0, sig_list)), key=lambda x: -abs(x[1]))

keywords = list(map(lambda x: x[0], sig_list_sorted[:150]))
f = open("freq_keywords", "wt")
f.write(",".join(keywords))
f.close()

sorted(sig_list, key=lambda x: -x[1])

1.044604617152408 / (NT + T)

nt, t = 0, 5
(nt / NT - t / T) / (nt + t) * (NT + T)