from locale import normalize
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

"""
Imports data from spotify_songs.csv and splits desired data into train and test sets
"""

def getData():
    #Import csv and get desired columns  
    data = pd.read_csv('Spotify_Songs_Dataset.csv', sep=',')
    data = data[["danceability", "energy" ,"key", "mode" ,"speechiness" ,"acousticness" ,"instrumentalness" ,"liveness" ,"valence" ,"tempo" , "duration_ms", "playlist_genre"]]

    #Split into train and test
    trainx, testx, trainy, testy = train_test_split(
        data[["danceability", "energy" ,"key", "mode" ,"speechiness" ,"acousticness" ,"instrumentalness" ,"liveness" ,"valence" ,"tempo", "duration_ms"]], 
        data[["playlist_genre"]], 
        test_size=0.2)

    #Convert to numpy arrays
    trainx = trainx.to_numpy()
    testx = testx.to_numpy()
    trainy = trainy.to_numpy()
    testy = testy.to_numpy()

    #Scale data
    scaler = preprocessing.StandardScaler()
    trainx = scaler.fit_transform(trainx)
    testx = scaler.fit_transform(testx)

    #Encode categorical features
    enc = preprocessing.OneHotEncoder()
    trainy = enc.fit_transform(trainy).toarray()
    testy = enc.fit_transform(testy).toarray()

    return trainx, testx, trainy, testy

if __name__ == '__main__':
    getData()