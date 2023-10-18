import tensorflow as tf
from tensorflow import keras
import numpy as np
import scipy
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys


from model import ResNN
from config.pure_brownian import f_func,b_func,G_func,sigma_func
from config.vanilla_call import g_func,d_g_func
from utils.parser import parser


class temporal_difference_learning():
    def __init__(self,T,N,batch_size,d_var,d_hidden,G_func,b_func,sigma_func,f_func,g_func,d_g_func,poisson_lambda):
        """
        T:
        N:
        batch_size:
        max_iter:
        G_func:
        b_func:
        sigma_func:
        poisson_lambda: 
        """
        self.T = T
        self.N = N
        self.batch_size = batch_size
        self.d_var = d_var
        # generate random sample
        self.dt = T/N
        self.t = tf.range(0,T+self.dt,self.dt)
        self.X = np.random.randn(batch_size,N+1,d_var)
        self.poisson_lambda = poisson_lambda

        # functional component
        self.b_func = b_func
        self.sigma_func = sigma_func
        self.G_func = G_func
        self.f_func = f_func
        self.g_func = g_func
        self.d_g_func = d_g_func

        # neural network
        self.model = ResNN(d_hidden=d_hidden)

        # optimizer
        # adam optimizer with lr_0 = 5e-5 and decay by 0.2 every 100 iterations
        lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
            [100,200,300,400], [5e-5,1e-5,2e-6,4e-7,8e-8])
        self.optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    
    def train_loop(self):
        # sample Brownian motion
        self.dW = scipy.stats.norm.rvs(scale=math.sqrt(self.dt),size=(self.batch_size,self.N,self.d_var))
        # sample jumps
        # TODO: sample in batch_size
        # E = []
        # while sum(E) < self.T:
        #     E.append(scipy.stats.expon.rvs(scale=1/self.poisson_lambda))   # sample exponential as arrival time
        # m = len(E) - 1
        # # four kinds of jump size distribution are considered in paper
        # J = scipy.stats.norm.rvs(loc=0.4,scale=0.25,size=m) # normal
        # J = scipy.stats.uniform.rvs(loc=-0.4,scale=0.8,size=m) # uniform
        # J = scipy.stats.expon.rvs(scale=1/3,size=m) # exponential
        # J = scipy.stats.bernoulli(0.7) * (-0.6) + 0.4

        for n in range(self.N-1):
            # eq 2.9
            # dimensions:
            # self.X[:,n,:]: [bs,d]
            # self.sigma_func(self.X[:,n,:]): [bs,d,d]
            # self.dW[:,n,:]: [bs,d,d]
            # self.dW[:,n,:]: [bs,d]
            self.X[:,n+1,:] = self.X[:,n,:] + self.b_func(self.X[:,n,:]) * self.dt + tf.einsum('bij,bj->bi',self.sigma_func(self.X[:,n,:]),self.dW[:,n,:]) # TODO: jump component
            with tf.GradientTape(persistent=True) as tape:
                X = tf.Variable(self.X,dtype=tf.float32)
                output = self.model(self.t,X) # output shape [bs,N+1,2]
                N1 = output[:,n,0]; N2 = output[:,n,1] # [bs,1]
                dN1 = tape.gradient(N1,X)[:,n] # [bs,d]
                N1_T = output[:,-1,0] # [bs,1]
                dN1_T = tape.gradient(N1_T,X)[:,-1] # [bs,d]
                N1_next = output[:,n+1,0]

                # compute loss
                # dimensions:
                # self.X[:,n,:]: [bs,d]
                # self.sigma_func(self.X[:,n,:]): [bs,d,d]
                # self.dW[:,n,:]: [bs,d,d]
                # self.dW[:,n,:]: [bs,d]
                TD_error = -self.f_func(self.X[:,n,:]) * self.dt + tf.einsum('bi,bi->b',tf.einsum('bji,bj->bi',self.sigma_func(self.X[:,n,:]),dN1),self.dW[:,n]) - self.dt * N2 + N1 - N1_next
                Loss_1 = tf.linalg.norm(TD_error,ord=2)/self.batch_size
                Loss_2 = tf.linalg.norm(N1_T - self.g_func(self.X[:,self.N,:]),ord=2)/(self.N * self.batch_size)
                Loss_3 = tf.linalg.norm(dN1_T - self.d_g_func(self.X[:,self.N,:]),ord=2)/(self.N * self.batch_size)
                Loss_4 = tf.linalg.norm(- self.dt * N2,ord=2)/self.batch_size
                Loss = Loss_1 + Loss_2 + Loss_3 + Loss_4
                print(Loss)
                self.optimizer.minimize(Loss,self.model.trainable_weights,tape=tape)
                del tape



if __name__ == "__main__":

    # tf.compat.v1.disable_eager_execution()
    num_threads = 5
    os.environ["OMP_NUM_THREADS"] = "5"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "5"
    os.environ["TF_NUM_INTEROP_THREADS"] = "5"

    tf.config.threading.set_inter_op_parallelism_threads(
        num_threads
    )
    tf.config.threading.set_intra_op_parallelism_threads(
        num_threads
    )
    tf.config.set_soft_device_placement(True)

    args = parser.parse_args()

    Trainer = temporal_difference_learning(T=args.T,
                                           N=args.N,
                                           batch_size=args.batch_size,
                                           d_var=args.d_var,
                                           d_hidden=args.d_hidden,
                                           f_func=f_func,
                                           b_func=b_func,
                                           G_func=G_func,
                                           sigma_func=sigma_func,
                                           poisson_lambda=args.poisson_lambda,
                                           g_func=g_func,
                                           d_g_func=d_g_func)
    for i in range(400):
        Trainer.train_loop()
