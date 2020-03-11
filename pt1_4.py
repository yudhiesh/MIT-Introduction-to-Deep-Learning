import tensorflow as tf
import mitdeeplearning as mdl
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model


#Gradient computation with GradientTape
# y =x^2
x = tf.Variable(3.0)

#Initiate the gradient tape
with tf.GradientTape() as tape:
    #define the function 
    y = x * x
#Access the gradient -- derivative of y with respect to x
dy_dx = tape.gradient(y,x)
assert dy_dx.numpy() == 6.0