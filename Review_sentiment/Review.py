
import nltk
from nltk.classify.naivebayes import NaiveBayesClassifier
from nltk.classify.util import apply_features,accuracy
from nltk.probability import FreqDist, ELEProbDist

def get_words_in_reviews(reviews):
    all_words = []
    for (words) in reviews:
      all_words.extend(words)
    return all_words


def get_word_features(wordlist):
    wordlist = FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features


def read_reviews(fname, t_type):
    reviews = []
    f = open(fname, 'r', encoding="utf8")
    line = f.readline()
    while line != '':
        reviews.append([line, t_type])
        line = f.readline()
    f.close()
    return reviews


def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
      features['contains(%s)' % word] = (word in document_words)
    return features


def classify_tweet(tweet):
    return \
        classifier.classify(extract_features(nltk.word_tokenize(tweet)))


# read in postive and negative training reviews
pos_reviews = read_reviews('happy.txt', 'positive')
neg_reviews = read_reviews('sad.txt', 'negative')


# filter away words that are less than 3 letters to form the training data
reviews = []
for (words, sentiment) in pos_reviews + neg_reviews:
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
    reviews.append((words_filtered, sentiment))


# extract the word features out from the training data
word_features = get_word_features(\
                    get_words_in_reviews(reviews))


# get the training set and train the Naive Bayes Classifier
training_set = nltk.classify.util.apply_features(extract_features, reviews)
classifier = NaiveBayesClassifier.train(training_set)


# read in the test reviews and check accuracy
# to add your own test reviews, add them in the respective files
test_reviews = read_reviews('happy_test.txt', 'positive')
test_reviews.extend(read_reviews('sad_test.txt', 'negative'))
total = accuracy = float(len(test_reviews))

for tweet in test_reviews:
    if classify_tweet(tweet[0]) != tweet[1]:
        accuracy -= 1

      
tweet = 'sad'

print(classifier.classify(extract_features(tweet.split())))


print('Total accuracy: %f%% (%d/20).' % (accuracy / total * 100, accuracy))
