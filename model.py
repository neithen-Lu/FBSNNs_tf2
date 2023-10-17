"""
Based on FBSNNs written by Maziar Raissi
"""

import numpy as np
import tensorflow as tf

from tensorflow.keras import layers

class ResBlock(layers.Layer):
    """ Residual block containing 2 * (linear layer + activation) """
    def __init__(self, d_hidden):
        super(ResBlock, self).__init__()
        # Dimensionality of the two hidden layers
        self.d_hidden = d_hidden
    
        self.linear1 = layers.Dense(self.d_hidden,activation='tanh')
        self.linear2 = layers.Dense(self.d_hidden,activation='tanh')
    
    def call(self, x):
        # Computing the first fully connected output
        fx = self.linear1(x)
        # Computing the second fully connected output
        fx = self.linear2(fx)
        return fx + x # residual connection

class ResNN(layers.Layer):
    """ Residual network in Fig.1 """
    def __init__(self, d_hidden):
        super(ResNN, self).__init__()
        self.d_hidden = d_hidden
        self.linear_in = layers.Dense(self.d_hidden)
        # Five residual blocks, with hidden dimension d_hidden
        self.res_block1 = ResBlock(self.d_hidden)
        self.res_block2 = ResBlock(self.d_hidden)
        self.res_block3 = ResBlock(self.d_hidden)
        self.res_block4 = ResBlock(self.d_hidden)
        self.res_block5 = ResBlock(self.d_hidden)
        # Output linear layer
        self.linear_out = layers.Dense(2)

    # def build(self):
    #     super(ResNN, self).__init__()
    #     # Input linear layer
    #     # self.linear_in = layers.Dense(self.d_hidden)
    #     # Five residual blocks, with hidden dimension d_hidden
    #     self.res_block1.build()
    #     self.res_block2.build()
    #     self.res_block3.build()
    #     self.res_block4.build()
    #     self.res_block5.build()
        # Output linear layer
        # self.linear_out = layers.Dense(2)
    
    # @tf.function
    def call(self, inputs):
        # Computing the output of the input linear layer 
        fx = self.linear_in(inputs)
        # Computing the output of the five residual blocks
        fx = self.res_block1(fx)
        fx = self.res_block2(fx)
        fx = self.res_block3(fx)
        fx = self.res_block4(fx)
        fx = self.res_block5(fx)
        # Computing the output of the output linear layer 
        fx = self.linear_out(fx)
        return fx
