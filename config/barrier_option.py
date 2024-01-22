import tensorflow as tf
import numpy as np

STRIKE_PRICE = 0
BARRIER = 1.2

# up and out 
def barrier_fn(past_state,x):
    """
    [bs,1] -> [bs,1]
    """
    return np.logical_not(np.logical_or(np.logical_not(past_state),x>BARRIER))

# jump diffusion
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
    value = (np.squeeze(x) - STRIKE_PRICE).clip(min=0)
    return np.squeeze(np.expand_dims(value,1) * state)

def d_g_func(x,state):
    """
    [bs,1] -> [bs,1]
    """
    value = np.ones(x.shape) * (x>STRIKE_PRICE)
    return value * state

# true price function
def u_exact(t,T,X,state): # (N+1) x 1, (N+1) x D
    v = (X-STRIKE_PRICE).clip(min=0)
    return v * (1-np.exp(-2* (np.log(BARRIER/X)**2) / (0.4**2 * (T-t)))) * state