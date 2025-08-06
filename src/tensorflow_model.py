import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def create_and_train_nn(input_dim, output_dim, X, y, epochs=1000):
    model = Sequential()
    model.add(Dense(1, input_dim=input_dim, activation='sigmoid'))  # Cause for here AND and OR are linear seperable : For AND, OR (1 layer)
    model.compile(optimizer=Adam(learning_rate=0.5), loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X, y, epochs=epochs, verbose=0)
    return model

def create_and_train_xor_nn(X, y, epochs=100):
    model_xor = Sequential()
    model_xor.add(Dense(2, input_dim=2, activation='sigmoid'))  # Cause I am using XOR here so : Hidden layer with 2 units
    model_xor.add(Dense(1, activation='sigmoid'))  # Output layer
    model_xor.compile(optimizer=Adam(learning_rate=0.5), loss='binary_crossentropy', metrics=['accuracy'])

    model_xor.fit(X, y, epochs=epochs, verbose=0)
    return model_xor
