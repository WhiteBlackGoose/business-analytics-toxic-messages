import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords


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

stopwords = set(stopwords.words("english"))

def remove_stopwords(words):
    wordsFiltered = []
    for w in words:
        if w not in stopwords:
            wordsFiltered.append(w)
    return wordsFiltered

def proc(text, ns):
    text = text.lower()
    text = remove_punct(text)
    text = remove_shorts(text)
    text = word_tokenize(text)
    text = remove_stopwords(text)
    ngrams = []
    for n in ns:
        for w in range(len(text) - n + 1):
            ngrams.append(" ".join(text[w:w+n]))
    return ngrams

# nltk.download('punkt')
# nltk.download('stopwords')
