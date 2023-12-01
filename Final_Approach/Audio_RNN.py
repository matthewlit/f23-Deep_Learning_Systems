import json
import numpy as np
import matplotlib.pyplot as plt
from keras import layers, models, losses, optimizers
from sklearn.model_selection import train_test_split

DATA_PATH = "mfcc_dataset.json"

#AVG Test accuracy:79% 
#Highest Test Accuracy 97%
#Validation accuracy:79%

def get_data(data_path):
    #get mfcc data from json file
    with open(data_path, "r") as file_path:
        data = json.load(file_path)

    x = np.array(data["mfcc"])
    y = np.array(data["labels"])    

    return x, y

def prep_data(test_size, validation_size):
    #split dataset into training, test, and validation set
    x, y = get_data(DATA_PATH)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=validation_size)

    return x_train, x_validation, x_test, y_train, y_validation, y_test

def build_model(input_shape):
    #build RNN model  
    model = models.Sequential()

    #LSTM layer
    model.add(layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(layers.LSTM(64))

    # dense layer
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))

    #softmax layer for output
    model.add(layers.Dense(10, activation='softmax'))

    return model

def plot_history(history):
    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error subplot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()

if __name__ ==  "__main__":
    #get data splits
    x_train, x_validation, x_test, y_train, y_validation, y_test = prep_data(0.25, 0.2)

    #build input and model
    input_shape = (x_train.shape[1], x_train.shape[2])
    model = build_model(input_shape)

    #compile model and tune learning rate
    optimizer = optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, 
                  loss=losses.SparseCategoricalCrossentropy(), 
                  metrics=['accuracy'])
    
    #print model summary
    model.summary()

    #graphs training and test accuracy and error
    history = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), batch_size=32, epochs=60)
    plot_history(history)

    #prints test and validation accuracy
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    print("\nTest accuracy: ", test_accuracy)
    print("Test loss: ", test_loss)
    