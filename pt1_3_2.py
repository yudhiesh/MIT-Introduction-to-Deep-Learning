import tensorflow as tf
import mitdeeplearning as mdl
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


### Defining a neural network using the Sequential API ###
n_output_nodes = 3
# N of outputs
# Define the model
model = Sequential()
dense_layer = Dense(n_output_nodes,activation='sigmoid')
model.add(dense_layer)
# test model with example output
x_input = tf.constant([[1,2.]], shape=(1,2))

model_output = model(x_input).numpy()
print(model_output)
# [[0.11664501 0.68260515 0.7555148 ]]
