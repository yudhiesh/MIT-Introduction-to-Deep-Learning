import tensorflow as tf
import mitdeeplearning as mdl
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model


class SubclassModel(tf.keras.Model):
    def __init__(self, n_output_nodes):
        super(SubclassModel, self).__init__()
        self.dense_layer = Dense(n_output_nodes, activation='sigmoid')

    def call(self, inputs):
        return self.dense_layer(inputs)


n_output_nodes = 3
model = SubclassModel(n_output_nodes)

x_input = tf.constant([[1, 2.]], shape=(1, 2))

print(model.call(x_input))

# tf.Tensor([[0.2597089  0.62293065 0.37206978]], shape=(1, 3), dtype=float32)