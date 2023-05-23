import pandas as pd
pd.set_option('display.max_columns', None)

train = pd.read_csv("./train.csv")[['target', 'comment_text']][:10000]
train['toxic'] = 1 * (train['target'] > 0.5)
# test = pd.read_csv("./test.csv")[['comment_text']]

unique_words = set()

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk

def remove_punct(text):
    import string
    punctuations = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'  # \' removed
    text = ' '.join(filter(None, (word.strip(string.punctuation) for word in text.split())))
    return text

def remove_shorts(text):
    return text\
        .replace("'s", "")\
        .replace("'re", "")\
        .replace("'d", "")\
        .replace("'ve", "")\
        .replace("'ll", "")\
        .replace("n't", " not")

def remove_stopwords(stopwords, words):
    wordsFiltered = []
    for w in words:
        if w not in stopWords:
            wordsFiltered.append(w)
    return wordsFiltered

def proc(stopwords, text, ns):
    text = text.lower()
    text = remove_punct(text)
    text = remove_shorts(text)
    text = word_tokenize(text)
    text = remove_stopwords(stopwords, text)
    ngrams = []
    for n in ns:
        for w in range(len(text) - n + 1):
            ngrams.append(" ".join(text[w:w+n]))
    return ngrams

#nltk.download('punkt')
#nltk.download('stopwords')
stopWords = set(stopwords.words('english'))

ex = (train['comment_text'][0]).lower()
ex = remove_punct(ex)
ex = remove_shorts(ex)
words = word_tokenize(ex)
words = remove_stopwords(stopwords, words)

words

from tqdm import tqdm

all_words = set()
for text in tqdm(train['comment_text']):
    all_words = all_words.union(proc(stopwords, text, [1, 2]))

len(all_words)
import seaborn as sns
import matplotlib.pyplot as plt
sns.histplot(train['target'][:2000])
plt.savefig('tmp/aaa.png')

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
    words = proc(stopwords, text, [1, 2])
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



len(sig_list_sorted[:10])
sig_list_sorted = sorted(sig_list, key=lambda x: -abs(x[1]))

sorted(sig_list, key=lambda x: -x[1])

1.044604617152408 / (NT + T)

nt, t = 0, 5
(nt / NT - t / T) / (nt + t) * (NT + T)