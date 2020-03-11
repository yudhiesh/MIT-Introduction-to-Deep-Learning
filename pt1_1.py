import numpy as np
import tensorflow as tf

sport = tf.constant("Tennis", tf.string)
number = tf.constant(1.414213563626, tf.float64)
# 0-d Tensors
print("`sport` is a {}-d Tensor".format(tf.rank(sport).numpy()))
print("`number `is a {}-d tensor".format(tf.rank(number).numpy()))
# 1-d Tensors
sport1 = tf.constant(["Tennis", "Basketball"], tf.string)
number1 = tf.constant([3.141592, 1.414213, 2.71821], tf.float64)
print("`sport` is a {}-d Tensor".format(tf.rank(sport1).numpy(), tf.shape(sport1)))
print("`number `is a {}-d tensor".format(tf.rank(number1).numpy(), tf.shape(number1)))

# 2-d Tensors
sport2 = tf.constant([["Tennis", "Football"]], tf.string)
number2 = tf.constant([[3.2332322, 2.232323323]], tf.float64)
print("`sport` is a {}-d Tensor".format(tf.rank(sport2).numpy(), tf.shape(sport1)))
print("`number `is a {}-d tensor".format(tf.rank(number2).numpy(), tf.shape(number1)))
assert isinstance(sport2, tf.Tensor), "matrix must be a tf Tensor object"
assert tf.rank(sport2).numpy() == 2

images = tf.zeros([10,256,256,3], tf.int32)
assert isinstance(images, tf.Tensor), "matrix must be a tf Tensor object"
assert tf.rank(images).numpy() == 4, "matrix must be of rank 4"
assert tf.shape(images).numpy().tolist() == [10, 256, 256, 3], "matrix is incorrect shape"
