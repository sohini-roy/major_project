import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

test = pd.read_csv("testing_set_data.csv", header=0, delimiter="\t", \
                   quoting=3 )


def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review, "html5lib").get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return(" ".join(meaningful_words))



forest_pickle = open('model.pickle','rb')
forest = pickle.load(forest_pickle)
vectorizer_pickle = open('vectorizer.pickle','rb')
vectorizer = pickle.load(vectorizer_pickle)

num_reviews = len(test["review"])
clean_test_reviews = []
print("Cleaning and parsing the test set movie reviews...\n")
for i in range(0,num_reviews):
    if( (i+1) % 1000 == 0 ):
        print("Review %d of %d\n" % (i+1, num_reviews))
    clean_review = review_to_words( test["review"][i] )
    clean_test_reviews.append( clean_review )

print(len(clean_test_reviews))
print(num_reviews)
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()
# print(test_data_features)
result = forest.predict(test_data_features)
# print(result)
output = pd.DataFrame( data={"id":test["id"], "actual_sentiment":test["sentiment"], "predicted_sentiment":result} )
# print(output)
output.to_csv( "Bag_of_Words_model.csv", index=False, sep='\t', quoting=3,  )

compare_data = pd.read_csv("Bag_of_Words_model.csv", header=0, delimiter="\t", \
                   quoting=3 )

num_data = compare_data["id"].size
count = 0
for i in range(0,num_data):
    if compare_data["actual_sentiment"][i] == compare_data["predicted_sentiment"][i]:
        count += 1

print("Count = "+str(count))
accuracy = (count/num_data)*100
print("Accuracy = "+str(accuracy)+"%")
