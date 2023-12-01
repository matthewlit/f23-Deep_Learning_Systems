import Spotify_Preprocess
import tensorflow as tf
from keras import layers, losses

"""
Convolutional Neural Network for Song Genre Classification Based on Spotify API Data
HIGH: 49% 
"""

def cnn():
    #Preprocess dataset for CNN
    print("Preprocessing...")
    trainx, testx, trainy, testy = Spotify_Preprocess.getData()

    #Implement CNN model
    print("Building Model...")
    input_shape = (trainx.shape[1],1)
    model = tf.keras.Sequential([
        layers.Conv1D(64, 2, activation='relu', input_shape=input_shape),
        layers.Conv1D(64, 2, activation='relu'),
        layers.MaxPooling1D(),
        layers.Conv1D(128, 2, activation='relu'),
        layers.Conv1D(128, 2, activation='relu'),
        layers.MaxPooling1D(),
        layers.Flatten(),
        layers.Dense(500, activation='relu'),
        layers.Dense(6, activation='softmax')
    ])

    #Compile CNN model
    model.compile(optimizer='Adam', loss=losses.CategoricalCrossentropy(), metrics=['accuracy'])

    #Train CNN model
    print("\nTraining...")
    model.fit(trainx, trainy, epochs=8)

    #Test CNN model
    print("\nTesting...")
    results = model.evaluate(testx, testy)
    print("Results [Loss, Accuracy]: ", results)

if __name__ == '__main__':
    cnn()