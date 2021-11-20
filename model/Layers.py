import tensorflow as tf
import math

from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Layer


class Conv2D_CD(Layer):
    def __init__(self, filters, kernel_size=3, strides=1,
                 padding='SAME', use_bias=False, theta=0.7):
        super(Conv2D_CD, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.theta = theta
        self.use_bias = use_bias
        self.kernel_initializer = VarianceScaling(scale=2.0)

    def build(self, input_shape):
        self._filter = tf.compat.v1.get_variable(name='conv2d_cd', shape=[
                                                 self.kernel_size, self.kernel_size, input_shape[-1], self.filters], initializer=self.kernel_initializer)

    def call(self, inputs):
        out_normal = tf.nn.conv2d(inputs, self._filter, strides=[
                                  1, self.strides, self.strides, 1], padding=self.padding, name='conv2d_cd/normal')
        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        kernel_diff = tf.math.reduce_sum(self._filter, axis=0, keepdims=True)
        kernel_diff = tf.math.reduce_sum(kernel_diff, axis=1, keepdims=True)
        kernel_diff = tf.tile(
            kernel_diff, [self.kernel_size, self.kernel_size, 1, 1])
        out_diff = tf.nn.conv2d(inputs, kernel_diff, strides=[
                                1, self.strides, self.strides, 1], padding=self.padding, name='conv2d_cd/diff')

        return out_normal - self.theta * out_diff
