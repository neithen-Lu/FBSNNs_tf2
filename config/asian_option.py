import tensorflow as tf
import numpy as np
import scipy

STRIKE_PRICE = 0

# geometric asian option
def f_func(x,poisson_lambda):
    """
    [bs,d] -> [bs]
    """
    return np.zeros(x.shape[0])

def G_func(x,z):
    return np.multiply(x,(np.exp(z)-1))

def b_func(x):
    """
    [bs,d] -> [bs,d]
    """
    return np.zeros(x.shape)

def sigma_func(x):
    """
    [bs,d] -> [bs,d,d]
    """
    return np.repeat(np.expand_dims(np.identity(x.shape[-1],dtype=np.float32) * 0.4,0),x.shape[0],axis=0)

def int_G_nv_dz_func(x,poisson_lambda,normal_mu,normal_sigma):
    return poisson_lambda * x * (np.exp(normal_mu+normal_sigma**2/2)-1)


def g_func(x,state):
    """
    [bs,d] -> [bs]
    if state == True, value = original value
    else value = 0
    """
    value = (state - STRIKE_PRICE).clip(min=0)
    return np.squeeze(value)

def d_g_func(x,state,sampling_freq):
    """
    [bs,1] -> [bs,1]
    """
    value = np.ones(x.shape) * (state>STRIKE_PRICE)
    return value / sampling_freq

# true price function
def u_exact(t,T,X,state): # (N+1) x 1, (N+1) x D
    T = T-t
    sigma_G = 0.4/np.sqrt(3); b = -sigma_G**2/4
    d1 = (np.log(X/STRIKE_PRICE) + (b+sigma_G**2/2)*T) / (np.sqrt(T)*sigma_G)
    d2  = d1 - np.sqrt(T)*sigma_G
    v = X * np.exp(b*T) * scipy.stats.norm.cdf(d1) - STRIKE_PRICE * scipy.stats.norm.cdf(d2)
    return v