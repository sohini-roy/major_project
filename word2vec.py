import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data
import logging
from gensim.models import word2vec


train = pd.read_csv( "labeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )
test = pd.read_csv( "testData.tsv", header=0, delimiter="\t", quoting=3 )
unlabeled_train = pd.read_csv( "unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )

print("Read %d labeled train reviews, %d labeled test reviews, and %d unlabeled reviews\n" % (train["review"].size, test["review"].size, unlabeled_train["review"].size ))


def review_to_wordlist( review, remove_stopwords=False ):
     review_text = BeautifulSoup(review, "html5lib").get_text()
     review_text = re.sub("[^a-zA-Z]"," ", review_text)
     words = review_text.lower().split()
     if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
     return(words)


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append( review_to_wordlist( raw_sentence, remove_stopwords ))
    return sentences


k=0
sentences = []
print("Parsing sentences from training set")
for review in train["review"]:
    k += 1
    sentences += review_to_sentences(review, tokenizer)
    if( (k) % 1000 == 0 ):
        print("train = "+str(k))

p=0
print("Parsing sentences from unlabeled set")
for review in unlabeled_train["review"]:
    p += 1
    sentences += review_to_sentences(review, tokenizer)
    if( (p) % 1000 == 0 ):
        print("unlabeled = "+str(p))


print(len(sentences))

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

num_features = 300    # Word vector dimensionality
min_word_count = 40   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

print("Training model...")
model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, min_count = min_word_count, window = context, sample = downsampling)
model.init_sims(replace=True)
model_name = "300features_40minwords_10context"
model.save(model_name)
