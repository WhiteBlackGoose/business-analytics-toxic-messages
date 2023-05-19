import pandas as pd
pd.set_option('display.max_columns', None)

train = pd.read_csv("./train.csv")[['target', 'comment_text']][:100000]
test = pd.read_csv("./test.csv")[['comment_text']]

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

nltk.download('punkt')
nltk.download('stopwords')
stopWords = set(stopwords.words('english'))

ex = (train['comment_text'][0]).lower()
ex = remove_punct(ex)
ex = remove_shorts(ex)
words = word_tokenize(ex)
words = remove_stopwords(stopwords, words)
