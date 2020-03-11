import tensorflow as tf
import mitdeeplearning as mdl
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model


class IdentityModel(tf.keras.Model):
    def __init__(self, n_output_nodes):
        super(IdentityModel, self).__init__()
        self.dense_layer = Dense(n_output_nodes, activation='sigmoid')

    def call(self, inputs, isidentity=False):
        x = self.dense_layer(inputs)
        if isidentity:
            return inputs
        return x


n_output_nodes = 3
model = IdentityModel(n_output_nodes)

x_input = tf.constant([[1, 2.]], shape=(1, 2))
'''TODO: pass the input into the model and call with and without the input identity option.'''
out_activate = model.call(x_input)  # TODO
# out_activate = # TODO
out_identity = model.call(x_input, isidentity=True)  # TODO
# out_identity = # TODO

print("Network output with activation: {}; network identity output: {}".format(
    out_activate.numpy(), out_identity.numpy()))

#Network output with activation: [[0.24436088 0.80442905 0.23763853]]; network identity output: [[1. 2.]]