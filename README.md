# Preprocessing

## Tokenization

File `./preprocess.py` exports functions to tokenize a tweet's text into most important n-grams

## Frequency

File `./preprocess_frequency.py` finds out the list of the words, which have the most effect on whether a tweet is considered toxic or not. It saves the found words into `./freq_keywords` (which must be the same among fit and production phases).

File `./freq.py` exports utilities to turn a given list of words into a vector.

## RNN

File `./rnn.py` exports types needed to work with RNN (such as `Encoder` and `Decoder`).

File `./preprocess_rnn.py` trains encoder and decoder together and saves encoder to files `./encoder.weights*`.

File ``

# Fitting

`./fit-svm.py` fits and tests SVM


# Results

Table with scores for different models

| Model             | Train Acc | Test Acc | Train F_1 | Test F_1 |
|-------------------|-----------|----------|-----------|----------|
| Freq SVM          | 0.9766    | 0.9568   | --        | --       |
| Freq SVM Balanced | 0.6919    | 0.6767   | 0.5939    | 0.5765   |
| RNN AdaBoost      | 0.9694    | 0.9614   | --        | --       |
