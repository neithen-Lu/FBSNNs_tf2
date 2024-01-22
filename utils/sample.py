import scipy
import math
import numpy as np


# sample Brownian motion
def sample_brownian(batch_size,N,d_var,dt):
    dW = scipy.stats.norm.rvs(scale=math.sqrt(dt),size=(batch_size,N,d_var))
    return dW


# sample jumps
def sample_jump(poisson_lambda,batch_size,T,N,d_var,dt,jump_type):
    poisson_exp_time = scipy.stats.expon.rvs(scale=1/poisson_lambda,size=(batch_size,1,1))
    while np.min(np.sum(poisson_exp_time,axis=1)) < T:
        # print(np.min(np.sum(exp_time,axis=1)))
        poisson_exp_time= np.concatenate((poisson_exp_time,scipy.stats.expon.rvs(scale=1/poisson_lambda,size=(batch_size,1,1))),axis=1)   # sample exponential as arrival time
    exp_time = (np.cumsum(poisson_exp_time,axis=1) // dt).astype(int) # discretize continuous jump times to time intervals
    jump_count = np.zeros((batch_size,N,1))
    jump_count[np.where(exp_time<N)[0],exp_time[exp_time<N]] = jump_count[np.where(exp_time<N)[0],exp_time[exp_time<N]]+1 
    # TODO ⬆️we currently ignore the possibility that multiple jumps may happen in the same time interval
    # jump size distribution 
    J = scipy.stats.norm.rvs(loc=0.4,scale=0.25,size=(batch_size,N,d_var)) # normal
    if jump_type == 'constant':
        J = np.ones((batch_size,N,d_var)) * 0.1
    J = np.multiply(jump_count,J)
    J = np.concatenate((J,np.zeros((batch_size,1,d_var))),axis=1)
    return J,poisson_exp_time