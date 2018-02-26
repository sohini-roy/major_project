import gensim
from gensim.models import Word2Vec
from gensim import corpora, models, similarities


# model = Word2Vec.load('word2vec_model')
model = Word2Vec.load('lyrics_word2vec_model')
print(model.doesnt_match("man woman child kitchen".split()))
# print(model.doesnt_match("paris berlin london austria".split()))
print(model.most_similar("man"))
# print(model.most_similar("queen"))
# print(model.most_similar("awful"))

# import pandas as pd
# from gensim.models import Word2Vec
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from bs4 import BeautifulSoup
# import re
# from nltk.corpus import stopwords
# import nltk.data
# import logging
#
#
# train = pd.read_csv( "labeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )
# model = Word2Vec.load("word2vec_model")
#
#
# def review_to_wordlist( review, remove_stopwords=False ):
#      review_text = BeautifulSoup(review, "html5lib").get_text()
#      review_text = re.sub("[^a-zA-Z]"," ", review_text)
#      words = review_text.lower().split()
#      if remove_stopwords:
#         stops = set(stopwords.words("english"))
#         words = [w for w in words if not w in stops]
#      return(words)
#
#
# def makeFeatureVec(words, model, num_features):
#      featureVec = np.zeros((num_features,),dtype="float32")
#      nwords = 0
#      index2word_set = set(model.index2word)
#      for word in words:
#         if word in index2word_set:
#             nwords = nwords + 1.
#             featureVec = np.add(featureVec,model[word])
#      featureVec = np.divide(featureVec,nwords)
#      return featureVec
#
#
# def getAvgFeatureVecs(reviews, model, num_features):
#     counter = 0
#     reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
#     for review in reviews:
#         if counter%1000 == 0:
#            print("Review %d of %d" % (counter, len(reviews)))
#         reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
#         counter = counter + 1
#     return reviewFeatureVecs
#
#
# num_features = 300    # Word vector dimensionality
# min_word_count = 40   # Minimum word count
# num_workers = 4       # Number of threads to run in parallel
# context = 10          # Context window size
# downsampling = 1e-3   # Downsample setting for frequent words
#
#
# clean_train_reviews = []
# for review in train["review"]:
#     clean_train_reviews.append( review_to_wordlist( review, remove_stopwords=True ))
#
# trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model, num_features )
#
# print("Creating average feature vecs for test reviews")
# clean_test_reviews = []
# for review in test["review"]:
#     clean_test_reviews.append( review_to_wordlist( review, \
#         remove_stopwords=True ))
#
# testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, num_features )
# forest = RandomForestClassifier( n_estimators = 100 )
#
# print("Fitting a random forest to labeled training data...")
# forest = forest.fit( trainDataVecs, train["sentiment"] )
# result = forest.predict( testDataVecs )
# output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
# output.to_csv( "Word2Vec_AverageVectors.csv", index=False, quoting=3 )
# print("done")
