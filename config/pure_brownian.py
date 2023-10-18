import numpy as np
import tensorflow as tf

def f_func(x):
    """
    [bs,d] -> [bs]
    """
    return 0.25 * np.linalg.norm(x,ord=2,axis=1) / x.shape[-1]

def G_func(x,z):
    return 0

def b_func(x):
    return 0.25 * x

def sigma_func(x):
    """
    [bs,d] -> [bs,d,d]
    """
    return np.repeat(np.expand_dims(np.identity(x.shape[-1],dtype=np.float32) * 0.5,0),x.shape[0],axis=0)