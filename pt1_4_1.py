import tensorflow as tf
import mitdeeplearning as mdl
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model


# Function minimization with automatic differentiation and SGD

# Initialize a random value for our initial x
x = tf.Variable([tf.random.normal([1])])
print("Initialising x={}".format(x.numpy()))

learning_rate = 1e-2  # Learning rate for SGD
history = []
# Define the target value
x_f = 4


for i in range(500):
    with tf.GradientTape() as tape:
        loss = (x - x_f) ** 2
    # loss minimization using gradient tape
    # compute the derivative of the loss with respect to x
    grad = tape.gradient(loss, x)
    new_x = x - learning_rate*grad  # sdg update
    x.assign(new_x)  # update the value of x
    history.append(x.numpy()[0])

# Plot the evolution of x as we optimise x_f!
plt.plot(history)
plt.plot([0, 500], [x_f, x_f])
plt.legend(('Predicted', 'True'))
plt.xlabel('Iteration')
plt.ylabel('x value')
plt.show()
