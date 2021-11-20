import tensorflow as tf
import numpy as np

from model.Layers import Conv2D_CD
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D, Input, Dense, Flatten, BatchNormalization, Activation, concatenate, Resizing, Flatten, Dropout


class CDCN(Model):
    def __init__(self, theta=0.7, input_shape=(256, 256, 3), use_nn=True, g_dropout=True, l_dropout=False, dropout_val=0.2):
        super(CDCN, self).__init__()
        self.use_nn = use_nn
        self.g_dropout = g_dropout
        self.l_dropout = l_dropout
        self.dropout_val = dropout_val

        self.Conv1 = self.getConvLayer(64, theta, input_shape)

        self.Block1 = self.getBlock(theta)

        self.Block2 = self.getBlock(theta)

        self.Block3 = self.getBlock(theta)

        self.Conv2 = self.getConvLayer(128, theta)

        self.Conv3 = self.getConvLayer(64, theta)

        self.Conv4 = self.getConvLayer(1, theta)

        self.downsample32x32 = Resizing(32, 32, interpolation='bilinear')

        if use_nn:
            self.nn = Sequential()
            self.nn.add(Flatten())
            self.nn.add(Dense(1, activation='sigmoid'))

    def getBlock(self, theta):
        Block = Sequential()

        Block.add(Conv2D_CD(128, padding='SAME', theta=theta))
        Block.add(BatchNormalization())
        Block.add(Activation('relu'))
        if self.l_dropout:
            Block.add(Dropout(self.dropout_val))

        Block.add(Conv2D_CD(196, padding='SAME', theta=theta))
        Block.add(BatchNormalization())
        Block.add(Activation('relu'))
        if self.l_dropout:
            Block.add(Dropout(self.dropout_val))

        Block.add(Conv2D_CD(128, padding='SAME', theta=theta))
        Block.add(BatchNormalization())
        Block.add(Activation('relu'))
        if self.l_dropout:
            Block.add(Dropout(self.dropout_val))

        Block.add(MaxPooling2D((3, 3), strides=2, padding='SAME'))

        if self.g_dropout:
            Block.add(Dropout(self.dropout_val))

        return Block

    def getConvLayer(self, nb_filters, theta, input_shape=None):
        conv = Sequential()
        if input_shape:
            conv.add(Input(input_shape))
        conv.add(Conv2D_CD(nb_filters, padding='SAME', theta=theta))
        conv.add(BatchNormalization())
        conv.add(Activation('relu'))
        if self.g_dropout:
            conv.add(Dropout(self.dropout_val))
        return conv

    def call(self, input_tensor):	    	# x [3, 256, 256]

        x = self.Conv1(input_tensor)

        x_Block1 = self.Block1(x)	    	    	# x [128, 128, 128]
        x_Block1_32x32 = self.downsample32x32(x_Block1)   # x [128, 32, 32]

        x_Block2 = self.Block2(x_Block1)	    # x [128, 64, 64]
        x_Block2_32x32 = self.downsample32x32(x_Block2)   # x [128, 32, 32]

        x_Block3 = self.Block3(x_Block2)	    # x [128, 32, 32]
        x_Block3_32x32 = self.downsample32x32(x_Block3)   # x [128, 32, 32]

        # x [128*3, 32, 32]   ## Changed axis 1 with axis 3
        x = concatenate(
            (x_Block1_32x32, x_Block2_32x32, x_Block3_32x32), axis=-1)

        x = self.Conv2(x)    # x [128, 32, 32]
        x = self.Conv3(x)    # x [64, 32, 32]
        x = self.Conv4(x)    # x [1, 32, 32]

        x = tf.squeeze(x, axis=-1)

        if self.use_nn:
            y_pred = self.nn(x)
            return y_pred
        else:
            return x
