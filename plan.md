# Plan
## 2023-05-10 together: deciding on metrics
- Fβ score
- P(actual 0 | predicted 0) * P(actual 1 | predicted 1)
#### Result: chosen metric
## 2023-06-03 preprocessing
### 2023-05-23 together: Frequency
#### Preprocess text
- Remove stop-words
- Ignore punctuation
- Get the core part of each word

#### Collect n-grams over the dataset
Consider different ns: 1, 2, 3
#### Build frequency dictionary
Not sure how it should look like yet
#### Result: function, which turns text into frequency dictionary
### 2023-06-03 together: RNN
#### Preprocess words with word2vec
Text -> list of vectors
#### Find a pre-fit RNN which extracts a vector out of list of vectors
Might also be worth trying fitting our own RNN
#### Convert all texts into vectors
Run the RNN over the lists of vectors to get a vector for each text
#### Result: function, which turns text into vector
## 2023-06-08 Fit models
### 2023-06-08 Ilya: 
#### KNN with Frequency preprocessing
- Analyze results
#### Logistic Regression with RNN preprocessing
- Analyze results
#### Result: analysis from the fitted models
### 2023-06-08 Lev: 
#### SVM with Frequency preprocessing
- Analyze results
#### CatBoost with RNN preprocessing
- Analyze results
#### Result: analysis from the fitted models
### 2023-06-08 Egor: 
#### LDA/QDA with Frequency preprocessing
- Analyze results
#### RandomForest with RNN preprocessing
- Analyze results
#### Result: analysis from the fitted models
## 2023-06-13 Endgame
### 2023-06-10 Ilya and Egor: compare results with other models and draw conclusions
Build a table comparing all 6 models
#### Result: table with models and explanation and choice of best models
### 2023-06-10 Lev: documentation
Build documentation, explaining how to use it
#### Result: markdown file with documentation on injecting the code into production
### 2023-06-13 together: report and presentation
#### Presentation
#### 3 reports
