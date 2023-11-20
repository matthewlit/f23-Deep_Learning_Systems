import pandas as pd
from sklearn.model_selection import train_test_split

"""
Imports data from spotify_songs.csv and splits desired data into train and test sets
"""

def getData():
    #Import csv and get desired columns  
    data = pd.read_csv('spotify_songs.csv', sep=',')
    data = data[["danceability", "energy" ,"key" ,"loudness" ,"mode" ,"speechiness" ,"acousticness" ,"instrumentalness" ,"liveness" ,"valence" ,"tempo" ,"playlist_genre"]]

    #Split into train and test
    trainx, testx, trainy, testy = train_test_split(
        data[["danceability", "energy" ,"key" ,"loudness" ,"mode" ,"speechiness" ,"acousticness" ,"instrumentalness" ,"liveness" ,"valence" ,"tempo"]], 
        data[["playlist_genre"]], 
        test_size=0.2)

    # print(trainx)
    # print(trainy)
    # print(testx)
    # print(testy)

    return trainx, testx, trainy, testy

if __name__ == '__main__':
    getData()