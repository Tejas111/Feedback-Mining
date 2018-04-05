'''
this code is one which detects the polarity of the reviews given by the student to the teacher. 
'''
import nltk

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
    
 



pos_reviews=[('Teaching is really good','good'), 
('Teaching is amazing','good'),
('I like to hear more from the teacher','good'),
('I would recommend the teacher','good'),
('She is awesome teacher like','good')]

neg_reviews=[('A bad teacher','bad'),
('Teacher is horrible','bad'),
('she is a headache','bad'),
('boring disgusting shit useless cruel jealous','bad'),
('she is an enemy ruthless heartless','bad')]

avg_reviews=[('A fine teacher','average'),
('Teacher is OK','average')]

reviews=[]
for(words,sentiment)in pos_reviews+neg_reviews+avg_reviews:
  words_filtered=[e.lower() for e in words.split() if len(e)>=3]
  reviews.append((words_filtered,sentiment))

test_pos_reviews=[('she is awesome','good'), 
('she is cool and best','good')]

test_neg_reviews=[('I do not like that man','bad'),
('she is too bad','bad'),
('teacher is too irresponsible','bad'),
('teacher is annoying','bad')]
test_avg_reviews=[('She is OK','average')]
test_reviews=[]
for(test_words,test_sentiment)in test_pos_reviews+test_neg_reviews+test_avg_reviews:
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

tweet = 'She is fine'
print(classifier.classify(extract_features(tweet.split())))

classifier.show_most_informative_features(5)


print(nltk.classify.util.accuracy(classifier,test_training_set))

