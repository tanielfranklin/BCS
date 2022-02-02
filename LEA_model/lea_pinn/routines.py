import tensorflow as tf
import numpy as np

#@tf.function
def get_abs_max_grad(grad):
    r=np.zeros((len(grad))).astype(np.float32)
    for i in range(len(grad)):
        r[i]=tf.math.reduce_max(tf.abs(grad[i]))
    return tf.math.reduce_max(r)
#@tf.function
def get_abs_mean_grad(grad):
    r=np.zeros((len(grad))).astype(np.float32)
    for i in range(len(grad)):
        r[i]=tf.math.reduce_mean(tf.abs(grad[i]))
    return tf.math.reduce_mean(r)
    




