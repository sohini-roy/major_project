import gensim
from gensim.models import Word2Vec
from gensim import corpora, models, similarities


model = Word2Vec.load('300features_40minwords_10context')
print(model.doesnt_match("man woman child kitchen".split()))
print(model.doesnt_match("france england germany berlin".split()))
print(model.doesnt_match("paris berlin london austria".split()))
print(model.most_similar("man"))
print(model.most_similar("queen"))
print(model.most_similar("awful"))
