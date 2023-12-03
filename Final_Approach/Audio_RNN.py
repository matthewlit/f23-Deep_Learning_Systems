import json
import numpy as np
from keras import layers, models, losses
from sklearn.model_selection import train_test_split

DATA_PATH = "mfcc_dataset.json"
"""
Recurrent  Neural Network for Song Genre Classification Based on .wav Song Files
AVG Test accuracy:79% 
Highest Test Accuracy 97%
Validation accuracy:79%
"""

def prep_data(test_size, validation_size):
    #Get mfcc data from json file
    with open("mfcc_dataset.json", "r") as file_path:
        data = json.load(file_path)
    x = np.array(data["mfcc"])
    y = np.array(data["labels"])       

    #Split dataset into training, test, and validation set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=validation_size)

    return x_train, x_validation, x_test, y_train, y_validation, y_test

def rnn():
    #Get data splits
    print("Preprocessing...")
    x_train, x_validation, x_test, y_train, y_validation, y_test = prep_data(0.25, 0.2)

    #Build RNN model
    input_shape = (x_train.shape[1], x_train.shape[2])  
    model = models.Sequential([
    #LSTM layer
    layers.LSTM(64, input_shape=input_shape, return_sequences=True),
    layers.LSTM(64),

    #Dense layer
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),

    #Softmax layer for output
    layers.Dense(10, activation='softmax')
    ])

    #Compile model
    model.compile(optimizer='Adam', loss=losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    #Train model
    print("\nTraining...")
    model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=60)

    #Test model
    print("\nTesting...")
    results = model.evaluate(x_test, y_test)
    print("Results [Loss, Accuracy]: ", results)

if __name__ ==  "__main__":
    rnn()
    