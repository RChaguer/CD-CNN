import tensorflow as tf
import numpy as np

from tensorflow.keras.losses import Loss, MeanSquaredError

def contrast_depth_conv(input, dilation_rate=1, op_name='contrast_depth'):
    kernel_filter_list = [
        [[1, 0, 0], [0, -1, 0], [0, 0, 0]], [[0, 1, 0], [0, -1, 0], [0, 0, 0]], [[0, 0, 1], [0, -1, 0], [0, 0, 0]],
        [[0, 0, 0], [1, -1, 0], [0, 0, 0]], [[0, 0, 0], [0, -1, 1], [0, 0, 0]],
        [[0, 0, 0], [0, -1, 0], [1, 0, 0]], [[0, 0, 0], [0, -1, 0], [0, 1, 0]], [[0, 0, 0], [0, -1, 0], [0, 0, 1]]
    ]
    
    kernel_filter = np.array(kernel_filter_list, np.float32)
    
    kernel_filter = np.expand_dims(kernel_filter, axis = -1)
    kernel_filter = kernel_filter.transpose([1, 2, 3, 0])
    kernel_filter_tf = tf.constant(kernel_filter, dtype=tf.float32)
    input = tf.expand_dims(input, axis=-1)

    if dilation_rate == 1:
        contrast_depth = tf.nn.conv2d(input, kernel_filter_tf, strides=[1, 1, 1, 1], padding='SAME', name=op_name)
    else:
        contrast_depth = tf.nn.atrous_conv2d(input, kernel_filter_tf,rate=dilation_rate, padding='SAME', name=op_name)

    return contrast_depth

def contrast_depth_loss(y_true, y_pred):
    contrast_pred = contrast_depth_conv(y_pred, 1, 'contrast_pred')
    contrast_true = contrast_depth_conv(y_true, 1, 'contrast_true')
    loss = tf.reduce_mean(tf.math.square(contrast_pred - contrast_true), axis=-1)
    return loss

class DepthLoss(Loss):
    def call(self, y_true, y_pred):
        mse = MeanSquaredError()
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        l1 = contrast_depth_loss(y_true, y_pred)
        l2 = mse(y_true, y_pred)
        return l1 + l2
    
def accuracy(y_true, y_pred):
    labels = tf.math.reduce_mean(y_true, axis=(1, 2), keepdims=True)
    l_pred = tf.math.reduce_mean(y_pred, axis=(1, 2), keepdims=True)
    return tf.keras.metrics.binary_accuracy(labels, l_pred)