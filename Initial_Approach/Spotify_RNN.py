import Spotify_Preprocess
import tensorflow as tf
from keras import layers, losses

"""
Recurrent Neural Network for Song Genre Classification Based on Spotify API Data
HIGH: 53% 
"""

def rnn():
    #Preprocess dataset for RNN
    print("\nPreprocessing...")
    trainx, testx, trainy, testy = Spotify_Preprocess.getData()

    #Implement RNN model
    print("\nBuilding Model...")
    input_shape = (trainx.shape[1],1)
    model = tf.keras.Sequential([
        layers.Bidirectional(layers.GRU(64, activation='relu', return_sequences=True), input_shape=input_shape),
        layers.Bidirectional(layers.GRU(128)),
        layers.Dense(128, activation='relu'),
        layers.Dense(6, activation='softmax')
    ])

    #Compile RNN model
    model.compile(optimizer='Adam', loss=losses.CategoricalCrossentropy(), metrics=['accuracy'])

    #Train RNN model
    print("\nTraining...")
    model.fit(trainx, trainy, epochs=8)

    #Test RNN model
    print("\nTesting...")
    results = model.evaluate(testx, testy)
    print("\nResults [Loss, Accuracy]: ", results)

if __name__ == '__main__':
    rnn()