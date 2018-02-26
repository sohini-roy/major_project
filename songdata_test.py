import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
import pickle


def testdata():
    test = pd.read_csv("songs_test_set_data.csv")


    def lyrics_to_words(raw_lyric):
        lyric_text = BeautifulSoup(raw_lyric, "html5lib").get_text()
        letters_only = re.sub("[^a-zA-Z]", " ", lyric_text)
        words = letters_only.lower().split()
        stops = set(stopwords.words("english"))
        meaningful_words = [w for w in words if not w in stops]
        return(" ".join(meaningful_words))


    forest_pickle = open('song_model.pickle','rb')
    forest = pickle.load(forest_pickle)
    vectorizer_pickle = open('song_vectorizer.pickle','rb')
    vectorizer = pickle.load(vectorizer_pickle)


    num_lyrics = len(test["Lyrics"])
    clean_test_lyrics = []
    # print("Cleaning and parsing the test set movie reviews...\n")
    for i in range(0,num_lyrics):
        # if( (i+1) % 10 == 0 ):
            # print("Lyric %d of %d\n" % (i+1, num_lyrics))
        clean_lyrics = lyrics_to_words( test["Lyrics"][i] )
        clean_test_lyrics.append( clean_lyrics )


    # print(len(clean_test_lyrics))
    test_data_features = vectorizer.transform(clean_test_lyrics)
    test_data_features = test_data_features.toarray()
    result = forest.predict(test_data_features)
    output = pd.DataFrame( data={"Title":test["Title"], "Actual_sentiment":test["Mood"], "Predicted_sentiment":result} )

    output.to_csv( "songdata_BOW_model.csv")

    compare_data = pd.read_csv("songdata_BOW_model.csv")

    num_data = compare_data["Title"].size
    count = 0
    for i in range(0,num_data):
        if compare_data["Actual_sentiment"][i] == compare_data["Predicted_sentiment"][i]:
            count += 1

    # print("Count = "+str(count))
    accuracy = (count/num_data)*100
    # print("Accuracy = "+str(accuracy)+"%")
    return accuracy
    # print("Data Tested")
