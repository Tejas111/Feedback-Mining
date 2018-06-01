
# This is used to determine the five different qualities of reviews and finally predicting those which are useful among them
from astropy.io import ascii
d = ascii.read('review.csv')
d.keys()
data = []
for row in d:
    if row['quality'] == 'awful':
        rating = 5
    elif row['quality'] == 'good':
        rating = 2
    elif row['quality'] == 'awesome':
        rating = 1
    elif row['quality'] == 'poor':
        rating= 4
    else:
        rating=3
    data.append((row['comment'], rating))

size = len(data)
r = int((size+1)*.7)
train = data[:r]
test = data[r:]

#Importing a model
from textblob.classifiers import NaiveBayesClassifier

cl = NaiveBayesClassifier(train)


cl.accuracy(test)

cl.show_informative_features(30)

db = ascii.read('Evaluations.csv')
hit, miss = 0, 0
for row in db:
    #print(cl.classify(row['col1']), row['col2'])
    if cl.classify(row['Review']) == row['Useful']:
        hit += 1
    else:
        miss += 1
#    for sentence in row['col1'].split('.'):
#        if cl.classify(sentence) == 0:
#            print(sentence)
#            print(cl.classify(sentence), row['col2'])       

print(hit, miss, hit/(hit+miss), miss/(hit+miss))

