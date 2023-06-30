# Preprocessing

[_**>> DATASET <<**_](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data)

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

| Model              | Train Acc | Test Acc | Train F_1 | Test F_1 |
|--------------------|-----------|----------|-----------|----------|
| Freq SVM           | 0.9766    | 0.9568   | 0.4392    | 0.2447   |
| RNN AdaBoost       | 0.9694    | 0.9614   | --        | --       |
| Freq LDA           | --        | --       | --        | 0.2400   |
| RNN Random Forest  | 0.9863    | 0.6000   | --        | 0.4440   |
| Freq KNN           | 0.9450    | 0.9434   | 0.3460    | 0.3090   |
| RNN Log Regression | 0.9700    | 0.9270   | --        | --       |
| Freq KNN Balanced  | 0.5270    | 0.5220   | 0.6590    | 0.6550   |
| Freq SVM Balanced  | 0.6919    | 0.6767   | 0.5939    | 0.5765   |
| Freq LDA Balanced  | 0.7930    | 0.7393   | --        | 0.6710   |


