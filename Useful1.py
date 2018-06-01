
# to find whether a given review is useful or not

# Neccesary Imports
from __future__ import print_function, division
import nltk
import os
import random
from collections import Counter
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import NaiveBayesClassifier, classify
import pandas as pd

# Initializations and functions
stoplist = stopwords.words('english')


def preprocess(sentence):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(str(sentence))]

def get_features(text, setting):
    if setting=='bow':
        return {word: count for word, count in Counter(preprocess(text)).items() if not word in stoplist}
    else:
        return {word: True for word in preprocess(text) if not word in stoplist}

def train(features, samples_proportion):
    train_size = int(len(features) * samples_proportion)
    # initialise the training and test sets
    train_set, test_set = features[:train_size], features[train_size:]
    print ('Training set size = ' + str(len(train_set)) + ' reviews')
    print ('Test set size = ' + str(len(test_set)) + ' reviews')
    # train the classifier
    classifier = NaiveBayesClassifier.train(train_set)
    return train_set, test_set, classifier

def evaluate(train_set, test_set, classifier):
    # check how the classifier performs on the training and test sets
    print ('Accuracy on the training set = ' + str(classify.accuracy(classifier, train_set)))
    print ('Accuracy of the test set = ' + str(classify.accuracy(classifier, test_set)))
    # check which words are most informative for the classifier
    classifier.show_most_informative_features(20)

# Reading in the corpus
df = pd.read_csv('useful.csv')
data = []
# arrange data in a tuple of the format (review,label)
for index,rows in df.iterrows():
    a = (rows['Review'],rows['Useful'])
    data.append(a)
# data
# for (each,label) in data:
#     print(each,label)


# Starting training and Evaluation

# feature extraction
corpus_features = [(get_features(each,''),label) for (each,label) in data]
print ('Collected ' + str(len(corpus_features)) + ' feature sets')

# training the classifier
train_set, test_set, classifier = train(corpus_features, 0.6)

# evaluate its performance
evaluate(train_set, test_set, classifier)



