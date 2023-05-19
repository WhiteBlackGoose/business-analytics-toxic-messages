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
        .replace("n't", " not")

def remove_stopwords(stopwords, words):
    wordsFiltered = []
    for w in words:
        if w not in stopWords:
            wordsFiltered.append(w)
    return wordsFiltered

def proc(stopwords, text):
    text = text.lower()
    text = remove_punct(text)
    text = remove_shorts(text)
    text = word_tokenize(text)
    text = remove_stopwords(stopwords, text)
    return text

# nltk.download('punkt')
# nltk.download('stopwords')
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
    all_words = all_words.union(proc(stopwords, text))

len(all_words)
import seaborn as sns
import matplotlib.pyplot as plt
sns.histplot(train['target'][:2000])
plt.savefig('/tmp/aaa.png')
# P(T | C) != P(!T | C)
#
#       T           !T
#
# C    100         1000
#
#
# !C    1            10
#


sum(train['toxic'])

s = dict()
for word in all_words:
    s[word] = [0, 0]

for text, tox in tqdm(zip(train['comment_text'], train['toxic'])):
    words = proc(stopwords, text)
    for word in words:
        s[word][tox] += 1

sig = dict()
NT = train.shape[0] - sum(train['toxic'])
T = sum(train['toxic'])
sig_list = []
for word in s:
    nt, t = s[word]
    if nt + t < 5:
        continue        
    sig[word] = abs(nt / NT - t / T)
    sig_list.append((word, abs(nt / NT - t / T)))

sig_list_sorted = sorted(sig_list, key=lambda x: -x[1])
