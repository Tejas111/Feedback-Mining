'''
this code is one which detects the polarity of the reviews given by the student to the teacher. 
'''
# This code gives another method to seperate reviews as useful or not
import nltk
import pandas as pd
from nltk.probability import FreqDist, ELEProbDist


from nltk.classify.util import apply_features,accuracy


def get_words_in_reviews(reviews):
    all_words = []
    for (words, sentiment) in reviews:
      all_words.extend(words)
    return all_words
    
def get_word_features(wordlist):
    wordlist = FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features
    
 



df = pd.read_csv('useful.csv')
data = []
# arrange data in a tuple of the format (review,label)
for index,rows in df.iterrows():
    a = (rows['Review'],rows['Useful'])
    data.append(a)
size = len(data)
r = int((size+1)*.7)
train = data[:r]
test = data[r:]
reviews=[]
for(words,sentiment)in train:
  words_filtered=[e.lower() for e in words.split() if len(e)>=3]
  reviews.append((words_filtered,sentiment))


test_reviews=[]
for(test_words,test_sentiment)in test:
  test_words_filtered=[e.lower() for e in test_words.split() if len(e)>=3]
  test_reviews.append((test_words_filtered,test_sentiment))
  
    
word_features = get_word_features(get_words_in_reviews(reviews))

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
      features['contains(%s)' % word] = (word in document_words)
    return features       
                       
    
training_set = apply_features(extract_features, reviews)

test_training_set=apply_features(extract_features, test_reviews)

classifier = nltk.classify.NaiveBayesClassifier.train(training_set)

tweet = 'she is awesome'
print(classifier.classify(extract_features(tweet.split())))

print(nltk.classify.util.accuracy(classifier,test_training_set))
classifier.show_most_informative_features(40)

