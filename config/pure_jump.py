import tensorflow as tf
import numpy as np

def f_func():
    return 0

def G_func(x,z):
    return x * (np.log(z)-1)

def b_func():
    return 0

def sigma_func():
    return 0