import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data
import logging
from gensim.models import Word2Vec


train = pd.read_csv("songs_train_set_data.csv")

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def lyric_to_wordlist( lyric, remove_stopwords=False ):
     lyric_text = BeautifulSoup(lyric, "html5lib").get_text()
     lyric_text = re.sub("[^a-zA-Z]"," ", lyric_text)
     words = lyric_text.lower().split()
     if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
     return(words)


def lyric_to_sentences( lyric, tokenizer, remove_stopwords=False ):
    raw_sentences = tokenizer.tokenize(lyric.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append( lyric_to_wordlist( raw_sentence, remove_stopwords ))
    return sentences


k=0
sentences = []
print("Parsing sentences from training set")
for lyric in train["Lyrics"]:
    k += 1
    sentences += lyric_to_sentences(lyric, tokenizer)
    if( (k) % 10 == 0 ):
        print("train = "+str(k))

print(len(sentences))

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

num_features = 300    # Word vector dimensionality
min_word_count = 40   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

print("Training model...")
model = Word2Vec(sentences, workers=num_workers, size=num_features, min_count = min_word_count, window = context, sample = downsampling)
model.init_sims(replace=True)
model_name = "lyrics_word2vec_model"
model.save(model_name)
