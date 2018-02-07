import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.model_selection import train_test_split


train = pd.read_csv("training_set_data.csv", header=0, \
                 delimiter="\t", quoting=3)



def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review, "html5lib").get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return(" ".join(meaningful_words))


# clean_review = review_to_words(train["review"][0])
# print(clean_review)

print("Cleaning and parsing the training set movie reviews...\n")
num_reviews = train["review"].size
clean_train_reviews = []
for i in range(0,num_reviews):
    if((i+1)%1000 == 0):
        print("Review %d of %d\n" % ( i+1, num_reviews ))
    clean_train_reviews.append(review_to_words(train["review"][i]))

print("Creating the bag of words...\n")
vectorizer = CountVectorizer(analyzer = "word",
                             tokenizer = None,
                             preprocessor = None,
                             stop_words = None,
                             max_features = 5000)
print(vectorizer)
train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()
print(train_data_features)
print(train_data_features.shape)

# vocab = vectorizer.get_feature_names()
# print(vocab)
# dist = np.sum(train_data_features, axis=0)
# print(dist)
# for tag, count in zip(vocab, dist):
#     print(count, tag)

print("Training the random forest...")
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit( train_data_features, train["sentiment"] )
print(forest)

pickle_model = open('model.pickle','wb')
pickle.dump(forest,pickle_model)
pickle_model.close()

vectorizer_file = open('vectorizer.pickle','wb')
pickle.dump(vectorizer,vectorizer_file)
vectorizer_file.close()
