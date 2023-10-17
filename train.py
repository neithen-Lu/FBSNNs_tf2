import tensorflow as tf
from tensorflow import keras
import numpy as np
import scipy
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys

# sess=tf.compat.v1.InteractiveSession()


from model import ResNN,ResBlock
from config.pure_brownian import f_func,b_func,G_func,sigma_func
from config.vanilla_call import g_func,d_g_func


class temporal_difference_learning():
    def __init__(self,T,N,batch_size,G_func,b_func,sigma_func,f_func,g_func,d_g_func,poisson_lambda):
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
        # generate random sample
        self.t = np.arange(T,N)
        self.dt = T/N
        self.X = np.random.randn(batch_size,N+1)
        self.poisson_lambda = poisson_lambda

        # functional component
        self.b_func = b_func
        self.sigma_func = sigma_func
        self.G_func = G_func
        self.f_func = f_func
        self.g_func = g_func
        self.d_g_func = d_g_func

        # neural network
        d_hidden = 25
        # self.model = tf.keras.Sequential([layers.Dense(d_hidden),ResBlock(d_hidden),ResBlock(d_hidden),ResBlock(d_hidden),ResBlock(d_hidden),ResBlock(d_hidden),layers.Dense(2)])
        self.model = ResNN(25)

        # optimizer
        # adam optimizer with lr_0 = 5e-5 and decay by 0.2 every 100 iterations
        lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
            [100,200,300,400], [5e-5,1e-5,2e-6,4e-7,8e-8])
        self.optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
    
    def train_loop(self):
        # sample Brownian motion
        self.dW = scipy.stats.norm.rvs(scale=math.sqrt(self.dt),size=(self.batch_size,self.N))
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

        # self.model.build(self.X.shape)
        # N = self.model.call(inputs=self.X)
        for n in range(self.N):
            # eq 2.9
            self.X[:,n+1] = self.X[:,n] + self.b_func(self.X[:,n]) * self.dt + self.sigma_func(self.X[:,n]) * self.dW[:,n] # TODO: jump component
            with tf.GradientTape() as tape:
                X = tf.expand_dims(tf.convert_to_tensor(self.X[:,n]),0)
                N = self.model(X)
                N1 = N[:,0]; N2 = N[:,1]
                # dN1 = tf.gradients(N1,X) 
                # print(dN1[0].eval(session=self.sess))
                # Loss_1 = -self.f_func(self.dt*n,self.X[:,n],N1,self.sigma_func(self.X[:,n])*dN1) * self.dt + tf.transpose(tf.transpose(self.sigma_func(self.X[:,n])) * dN1) * self.dW[:,n] - self.dt * N2 + N1 - self.model(self.X[:,n+1])[0]
                Loss_1 = 0
                X = tf.expand_dims(tf.convert_to_tensor(self.X[:,self.N]),0)
                N_T = self.model(X)
                N1_T = N_T[:,0]
                # dN1_T = tf.gradients(N1_T,X)[0]
                Loss_2 = (N1_T - self.g_func(self.X[:,self.N]))/self.N
                # Loss_3 = (dN1_T - self.d_g_func(self.X[:,self.N]))/self.N
                Loss_3 = 0
                Loss_4 = - self.dt * N2
                Loss = Loss_1 + Loss_2 + tf.cast(Loss_3, tf.float32) + Loss_4
                print(Loss.numpy())
                self.optimizer.minimize(Loss,self.model.trainable_weights,tape=tape)

    # def loss(self,n,t,X_t,N1,N2,dN1):
    #     Loss_1 = -self.f_func(t,X_t,N1,self.sigma_func(X_t)*dN1) * self.dt + tf.transpose(tf.transpose(self.sigma_func(X_t)) * dN1) * self.dW[:,n] - self.dt * N2 + N1 - self.model(self.X[:,n+1])[0]
    #     with tf.GradientTape() as tape:
    #         N1_T,_ = self.model(self.X[:,self.N])
    #     dN1_T = tape.gradient(N1_T, self.X[:,self.N]) 
    #     Loss_2 = (N1_T - self.g_func(self.X[:,self.N]))/self.N
    #     # Loss_3 = (dN1_T - self.d_g_func(self.X[:,self.N]))/self.N
    #     Loss_4 = - self.dt * N2
    #     return Loss_1 + Loss_2 + Loss_3 + Loss_4





            





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

    Trainer = temporal_difference_learning(T=1,N=24,batch_size=50,f_func=f_func,b_func=b_func,G_func=G_func,sigma_func=sigma_func,poisson_lambda=1,g_func=g_func,d_g_func=d_g_func)
    for i in range(100):
        Trainer.train_loop()
