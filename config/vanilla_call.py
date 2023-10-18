import numpy as np
import tensorflow as tf

def g_func(x):
    return np.linalg.norm(x,ord=2,axis=1)

def d_g_func(x):
    """
    [bs,d] -> [bs,d]
    """
    return np.ones(x.shape)