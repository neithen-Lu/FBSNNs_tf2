import tensorflow as tf
import numpy as np

# jump diffusion
def f_func(x):
    """
    [bs,d] -> [bs]
    """
    return np.zeros(x.shape[0])

def G_func(x,z):
    return np.multiply(x,(np.exp(z)-1))

def b_func(x):
    return np.zeros(x.shape)

def sigma_func(x):
    """
    [bs,d] -> [bs,d,d]
    """
    return np.repeat(np.expand_dims(np.identity(x.shape[-1],dtype=np.float32) * 0.4,0),x.shape[0],axis=0)

def int_G_nv_dz_func(x,poisson_lambda,normal_mu,normal_sigma):
    return poisson_lambda * x * (np.exp(normal_mu+normal_sigma**2/2)-1)

def g_func(x):
    return np.squeeze(x)

def d_g_func(x):
    """
    [bs,d] -> [bs,d]
    """
    return np.ones(x.shape)

# true price function
def u_exact(t,T,X): # (N+1) x 1, (N+1) x D
    return X