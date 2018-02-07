import pandas as pd
from sklearn.model_selection import train_test_split

train = pd.read_csv("labeledTrainData.tsv", header=0, \
                 delimiter="\t", quoting=3)

train_data, test_data = train_test_split(train, test_size=0.5)
train_data.to_csv( "training_set_data.csv", index=False, sep='\t', quoting=3 )
test_data.to_csv( "testing_set_data.csv", index=False, sep='\t', quoting=3 )
