## Functions to be imported in main.py
import tensorflow as tf
from tensorflow import keras
import pickle
import numpy as np
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Input, Model

# Helper function to load a pickle object in a more concise way.
def open_pickle(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

# Run the model naked, i.e. with no corrections. If this fails then there is
# an issue with the inputs. Also want to see what the bias is like here.

def naked_model(model_path, train_path, train_labels, test_path, test_labels):
    # Loading the simple model.
    model = keras.models.load_model(model_path)
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer = opt, loss = "binary_crossentropy", metrics = ['accuracy'])

    train_data = open_pickle(train_path)
    train_labels = open_pickle(train_labels)

    test_data = open_pickle(test_path)
    test_labels = open_pickle(test_labels)
    history = model.fit(train_data, train_labels, batch_size = 32, epochs = 5,\
     validation_split = 0.2, shuffle = True)

    eval = model.evaluate(test_data, test_labels)

    print("### BASE MODEL EVALUATION")
    for i in range(len(eval)):
        print(model.metrics_names[i] + ": " + str(eval[i]))


# Run the model with an adversarial component. Need to be given a class (str) that
# identifies how to run the adversary.
def combined_model(model_path, train_path, train_labels, test_path, test_label, class_name, alpha):
    # Need to modify the data based on input for class. Loading in training data.
    train_data = open_pickle(train_path)
    if class_name not in train_data.columns:
        print("Bad class name")
        return None
    Y_adv = train_data[class_name]
    train = train_data.drop([class_name], axis = 1)
    train_labels = open_pickle(train_labels)

    # Now loading the model
    model = keras.models.load_model(model_path)
    output_layer = model.layers[-1].output
    A1 = Dense(40, activation='relu',name='A1')(output_layer)
    A2 = Dense(1, activation='sigmoid',name='A2')(A1)

    combo_model = Model(inputs = [model.input], outputs = [output_layer , A2])
    print(combo_model.summary())
