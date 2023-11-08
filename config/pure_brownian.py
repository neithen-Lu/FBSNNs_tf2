import numpy as np
import tensorflow as tf

# pure brownian
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

def int_G_nv_dz_func(x,poisson_lambda,normal_mu,normal_sigma):
    return 0

def g_func(x):
    """
    [bs,d] -> [bs]
    """
    return np.linalg.norm(x,ord=2,axis=1)

def d_g_func(x):
    """
    [bs,d] -> [bs,d]
    """
    return np.ones(x.shape)

# true price function
def u_exact(t,T,X): # (N+1) x 1, (N+1) x D
    return np.exp(0.25*(T - t))*np.sqrt(np.sum(X**2, 1, keepdims = True))