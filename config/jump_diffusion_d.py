import tensorflow as tf
import numpy as np

# jump diffusion
def f_func(x,poisson_lambda):
    """
    [bs,d] -> [bs]
    """
    return - np.ones(x.shape[0]) * (poisson_lambda * 0.01 + 0.09)

def G_func(x,z):
    """
    [bs,d] -> [bs,d]
    """
    return z

def b_func(x):
    """
    [bs,d] -> [bs,d]
    """
    return np.zeros(x.shape)

def sigma_func(x):
    """
    [bs,d] -> [bs,d,d]
    """
    return np.repeat(np.expand_dims(np.identity(x.shape[-1],dtype=np.float32) * 0.3,0),x.shape[0],axis=0)

def int_G_nv_dz_func(x,poisson_lambda,normal_mu,normal_sigma):
    """
    [bs,d] -> [bs,d]
    """
    return np.ones(x.shape) * 0.1 * poisson_lambda

def g_func(x):
    """
    [bs,d] -> [bs]
    """
    return np.linalg.norm(x,ord=2,axis=1)**2 / x.shape[-1]

def d_g_func(x):
    """
    [bs,d] -> [bs,d]
    """
    return 2 * x / x.shape[-1]

# true price function
def u_exact(t,T,X): # (N+1) x 1, (N+1) x D
    return np.linalg.norm(X,ord=2,axis=1)**2 / X.shape[-1]