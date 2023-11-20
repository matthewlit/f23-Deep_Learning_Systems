import dataset

"""
Convolutional Neural Network for Song Genre Classification Based on Spotify API Data
"""

def cnn():
    #Preprocess dataset for CNN
    trainx, testx, trainy, testy = preprocess()

    #TODO: Implement CNN

    #TODO: Test CNN


#Preprocess dataset for CNN
def preprocess():
    #Get train and test dataset
    trainx, testx, trainy, testy = dataset.getData()

    #TODO: Preprocess for CNN

    return  trainx, testx, trainy, testy


if __name__ == '__main__':
    cnn()