import preprocess
import tensorflow as tf
from keras import layers, losses

"""
Recurrent Neural Network for Song Genre Classification Based on Spotify API Data
"""

def rnn():
    #Preprocess dataset for RNN
    print("\nPreprocessing...")
    trainx, testx, trainy, testy = preprocess.getData()

    #Implement RNN model
    print("\nBuilding Model...")
    input_shape = (trainx.shape[1],1)

    # TODO: Find best layer architecture for highest accuracy 
    # HIGH: 53% 
    model = tf.keras.Sequential([
        layers.Bidirectional(layers.GRU(64, activation='relu', return_sequences=True), input_shape=input_shape),
        layers.Bidirectional(layers.GRU(128)),
        layers.Dense(128, activation='relu'),
        layers.Dense(6, activation='softmax')
    ])

    model.compile(optimizer='Adam', loss=losses.CategoricalCrossentropy(), metrics=['accuracy'])

    print("\nTraining...")
    model.fit(trainx, trainy, epochs=8)

    #Test RNN
    print("\nTesting...")
    results = model.evaluate(testx, testy)
    print("\nResults [Loss, Accuracy]: ", results)

if __name__ == '__main__':
    rnn()