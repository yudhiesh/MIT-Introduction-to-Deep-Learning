import tensorflow as tf
import numpy as np

# Create nodes in a graph, and initialize values
a = tf.constant(15)
b = tf.constant(61)

# Add them
#c1 = tf.add(a,b)
#c2 = a + b 
#print(c1)
#print(c2)
def func(a,b):
    c = tf.add(a,b)
    d = tf.subtract(b,1)
    e = tf.multiply(c,d)
    return e

print(func(a,b))

