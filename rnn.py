import dataset

"""
Recurrent Neural Network for Song Genre Classification Based on Spotify API Data
"""

def rnn():
    #Preprocess dataset for RNN
    trainx, testx, trainy, testy = preprocess()

    #TODO: Implement RNN

    #TODO: Test RNN
    
    
#Preprocess dataset for RNN
def preprocess():
    #Get train and test datasets
    trainx, testx, trainy, testy = dataset.getData()

    #TODO: Preprocess for RNN

    return  trainx, testx, trainy, testy


if __name__ == '__main__':
    rnn()