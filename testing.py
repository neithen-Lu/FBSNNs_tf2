import tensorflow as tf
from tensorflow import keras
import numpy as np
import scipy
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import matplotlib.pyplot as plt


from model import ResNN
from config.pure_brownian import f_func,b_func,G_func,sigma_func,int_G_nv_dz_func
# from config.pure_jump import f_func,b_func,G_func,sigma_func,int_G_nv_dz_func
from config.vanilla_call import g_func,d_g_func
from utils.parser import parser

## 1. Test the neural network

class vanilla_NN():
    def __init__(self,T,N,batch_size,d_var,d_hidden):
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
        self.N = N
        self.batch_size = batch_size
        self.d_var = d_var
        self.dt = T/N
        self.t = tf.range(0,T+self.dt,self.dt)

        # neural network
        self.model = ResNN(d_hidden=d_hidden)

        # optimizer
        # adam optimizer with lr_0 = 5e-5 and decay by 0.2 every 100 iterations
        lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
            [100,200,300,400], [1e-4,1e-4,1e-5,1e-6,1e-7])
        self.optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    def train_loop(self,X,Y):
        for n in range(self.N):
            with tf.GradientTape(persistent=True) as tape:
                X = tf.Variable(X,dtype=tf.float32)
                output = self.model(self.t[n],X[:,n,:]) # output shape [bs,N+1,2]
                N1 = output[:,0]
                Loss = tf.linalg.norm(Y[:,n]-N1,ord=2)

                self.optimizer.minimize(Loss,self.model.trainable_weights,tape=tape)
                del tape
        print(Loss)



def test_1():
    Trainer = vanilla_NN(T=args.T,
                        N=args.N,
                        batch_size=args.batch_size,
                        d_var=args.d_var,
                        d_hidden=args.d_hidden)
    
    def u_exact(t, X): # (N+1) x 1, (N+1) x D
        r = 0.25
        sigma_max = 0.5
        return np.exp((r + sigma_max**2)*(args.T - t))*np.sqrt(np.sum(X**2, 1, keepdims = True))
    
    dt = args.T/args.N
    t = tf.range(0,args.T+dt,dt)
    t_test = np.repeat(np.expand_dims(t,0),args.batch_size,axis=0)
    for i in range(100):
        X = np.random.randn(args.batch_size,args.N+1,args.d_var)
        Y = np.reshape(u_exact(np.reshape(t_test,[-1,1]), np.reshape(X,[-1,args.d_var])),[args.batch_size,-1])
        Trainer.train_loop(X,Y)
    
    model = Trainer.model

    # test data
    X = np.random.randn(args.batch_size,args.N+1,args.d_var)
    Y_test = np.reshape(u_exact(np.reshape(t_test,[-1,1]), np.reshape(X,[-1,args.d_var])),[args.batch_size,-1])

    Y_pred = np.zeros(Y_test.shape)
    for n in range(args.N):
        output = model(t[n],X[:,n,:])
        Y_pred[:,n] = output[:,0].numpy()
    
    samples = 5
    
    plt.figure()
    print(t.shape,Y_test.shape)
    plt.plot(t,Y_pred[0,:].T,'b',label='Learned $u(t,X_t)$')
    plt.plot(t,Y_test[0,:].T,'r--',label='Exact $u(t,X_t)$')
    # plt.plot(t_test,Y_test[0,-1],'ko',label='$Y_T = u(T,X_T)$')
    
    # plt.plot(t_test,Y_pred[1:samples,:].T,'b')
    # plt.plot(t_test,Y_test[1:samples,:].T,'r--')
    # plt.plot(t_test,Y_test[1:samples,-1],'ko')

    # plt.plot([0],Y_test[0,0],'ks',label='$Y_0 = u(0,X_0)$')
    
    plt.xlabel('$t$')
    plt.ylabel('$Y_t = u(t,X_t)$')
    plt.title('100-dimensional Black-Scholes-Barenblatt')
    plt.legend()
    
    plt.savefig('test.png')

## 2. Test the 

def u_exact(t, X): # (N+1) x 1, (N+1) x D
    r = 0.25
    sigma_max = 0.5
    return np.exp((r + sigma_max**2)*(args.T - t))*np.sqrt(np.sum(X**2, 1, keepdims = True))

class brownian_NN():
    def __init__(self,T,N,batch_size,d_var,d_hidden):
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
        self.N = N
        self.batch_size = batch_size
        self.d_var = d_var
        self.dt = T/N
        self.t = tf.range(0,T+self.dt,self.dt)
        self.X = np.ones((args.batch_size,args.N+1,args.d_var)) * 0.1
        self.X[:,-1,:] = np.random.randn(args.batch_size,args.d_var)

        # neural network
        self.model = ResNN(d_hidden=d_hidden)

        # functional component
        self.b_func = b_func
        self.sigma_func = sigma_func
        self.G_func = G_func
        self.f_func = f_func
        self.g_func = g_func
        self.d_g_func = d_g_func

        # optimizer
        # adam optimizer with lr_0 = 5e-5 and decay by 0.2 every 100 iterations
        self.optimizer = keras.optimizers.Adam(learning_rate=5e-5)

    def train_loop(self):
        self.dW = scipy.stats.norm.rvs(scale=math.sqrt(self.dt),size=(self.batch_size,self.N,self.d_var))
        t_test = np.repeat(np.expand_dims(self.t,0),args.batch_size,axis=0)
        for n in range(self.N):
            self.X[:,n+1,:] = self.X[:,n,:] + self.b_func(self.X[:,n,:]) * self.dt + tf.einsum('bij,bj->bi',self.sigma_func(self.X[:,n,:]),self.dW[:,n,:])
            Y = np.reshape(u_exact(np.reshape(t_test,[-1,1]), np.reshape(self.X,[-1,args.d_var])),[args.batch_size,-1])
            with tf.GradientTape(persistent=True) as tape:
                X = tf.Variable(self.X,dtype=tf.float32)
                output = self.model(self.t[n],X[:,n,:]) # output shape [bs,N+1,2]
                N1 = output[:,0]
                Loss = tf.linalg.norm(Y[:,n]-N1,ord=2)

                self.optimizer.minimize(Loss,self.model.trainable_weights,tape=tape)
                del tape
        print(Loss)



def test_2():

    Trainer = brownian_NN(T=args.T,
                        N=args.N,
                        batch_size=args.batch_size,
                        d_var=args.d_var,
                        d_hidden=args.d_hidden)
    
    dt = args.T/args.N
    t = tf.range(0,args.T+dt,dt)
    t_test = np.repeat(np.expand_dims(t,0),args.batch_size,axis=0)
    for i in range(100):
        Trainer.train_loop()
    
    model = Trainer.model

    # test data
    X = np.ones((args.batch_size,args.N+1,args.d_var)) * 0.1
    dW = scipy.stats.norm.rvs(scale=math.sqrt(dt),size=(args.batch_size,args.N,args.d_var))
    for n in range(args.N):
        X[:,n+1,:] = X[:,n,:] + b_func(X[:,n,:]) * dt + tf.einsum('bij,bj->bi',sigma_func(X[:,n,:]),dW[:,n,:])
    Y_test = np.reshape(u_exact(np.reshape(t_test,[-1,1]), np.reshape(X,[-1,args.d_var])),[args.batch_size,-1])

    Y_pred = np.zeros(Y_test.shape)
    for n in range(args.N+1):
        output = model(t[n],X[:,n,:])
        Y_pred[:,n] = output[:,0].numpy()
    
    samples = 5
    
    plt.figure()
    print(t.shape,Y_test.shape)
    plt.plot(t,Y_pred[0,:].T,'b',label='Learned $u(t,X_t)$')
    plt.plot(t,Y_test[0,:].T,'r--',label='Exact $u(t,X_t)$')
    # plt.plot(t_test,Y_test[0,-1],'ko',label='$Y_T = u(T,X_T)$')
    
    # plt.plot(t_test,Y_pred[1:samples,:].T,'b')
    # plt.plot(t_test,Y_test[1:samples,:].T,'r--')
    # plt.plot(t_test,Y_test[1:samples,-1],'ko')

    # plt.plot([0],Y_test[0,0],'ks',label='$Y_0 = u(0,X_0)$')
    
    plt.xlabel('$t$')
    plt.ylabel('$Y_t = u(t,X_t)$')
    plt.title('100-dimensional Black-Scholes-Barenblatt')
    plt.legend()
    
    plt.savefig('test.png')


class identity_NN():
    def __init__(self,T,N,batch_size,d_var,d_hidden):
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
        self.N = N
        self.batch_size = batch_size
        self.d_var = d_var
        self.dt = T/N
        self.t = tf.range(0,T+self.dt,self.dt)
        self.X = np.ones((args.batch_size,args.N+1,args.d_var)) * 0.1
        self.X[:,-1,:] = np.random.randn(args.batch_size,args.d_var)

        # neural network
        self.model = tf.keras.layers.Identity()

        # functional component
        self.b_func = b_func
        self.sigma_func = sigma_func
        self.G_func = G_func
        self.f_func = f_func

        # optimizer
        # adam optimizer with lr_0 = 5e-5 and decay by 0.2 every 100 iterations
        self.optimizer = keras.optimizers.Adam(learning_rate=5e-5)

    def train_loop(self):
        self.dW = scipy.stats.norm.rvs(scale=math.sqrt(self.dt),size=(self.batch_size,self.N,self.d_var))
        t_test = np.repeat(np.expand_dims(self.t,0),args.batch_size,axis=0)
        for n in range(self.N):
            self.X[:,n+1,:] = self.X[:,n,:] + self.b_func(self.X[:,n,:]) * self.dt + tf.einsum('bij,bj->bi',self.sigma_func(self.X[:,n,:]),self.dW[:,n,:])
            Y = np.reshape(u_exact(np.reshape(t_test,[-1,1]), np.reshape(self.X,[-1,args.d_var])),[args.batch_size,-1])
            with tf.GradientTape(persistent=True) as tape:
                # X_n
                X_n = tf.Variable(self.X[:,n,:],dtype=tf.float32)
                output = self.model(self.t[n],X_n) # output shape [bs,2]
                N1 = output[:,0]; N2 = output[:,1] # [bs,]
                dN1 = tape.gradient(N1,X_n) # [bs,d]
                # X_T
                X_T = tf.Variable(self.X[:,-1,:],dtype=tf.float32)
                N1_T = self.model(self.t[-1],X_T)[:,0] 
                dN1_T = tape.gradient(N1_T,X_T) # [bs,d]
                # X_n+1
                X_next = tf.Variable(self.X[:,n+1,:],dtype=tf.float32)
                N1_next = self.model(self.t[n+1],X_next)[:,0]
                # jump output, used in Loss_1 and Loss_4
                X_jump = tf.Variable(self.X[:,n,:]+G_func(self.X[:,n,:],J[:,n,:]),dtype=tf.float32)
                N1_jump = self.model(self.t[n],X_jump)[:,0]

                Loss = tf.linalg.norm(Y[:,n]-N1,ord=2)

                self.optimizer.minimize(Loss,self.model.trainable_weights,tape=tape)
                del tape
        print(Loss)



def test_2():

    Trainer = brownian_NN(T=args.T,
                        N=args.N,
                        batch_size=args.batch_size,
                        d_var=args.d_var,
                        d_hidden=args.d_hidden)
    
    dt = args.T/args.N
    t = tf.range(0,args.T+dt,dt)
    t_test = np.repeat(np.expand_dims(t,0),args.batch_size,axis=0)
    for i in range(100):
        Trainer.train_loop()
    
    model = Trainer.model

    # test data
    X = np.ones((args.batch_size,args.N+1,args.d_var)) * 0.1
    dW = scipy.stats.norm.rvs(scale=math.sqrt(dt),size=(args.batch_size,args.N,args.d_var))
    for n in range(args.N):
        X[:,n+1,:] = X[:,n,:] + b_func(X[:,n,:]) * dt + tf.einsum('bij,bj->bi',sigma_func(X[:,n,:]),dW[:,n,:])
    Y_test = np.reshape(u_exact(np.reshape(t_test,[-1,1]), np.reshape(X,[-1,args.d_var])),[args.batch_size,-1])

    Y_pred = np.zeros(Y_test.shape)
    for n in range(args.N+1):
        output = model(t[n],X[:,n,:])
        Y_pred[:,n] = output[:,0].numpy()
    
    samples = 5
    
    plt.figure()
    print(t.shape,Y_test.shape)
    plt.plot(t,Y_pred[0,:].T,'b',label='Learned $u(t,X_t)$')
    plt.plot(t,Y_test[0,:].T,'r--',label='Exact $u(t,X_t)$')
    # plt.plot(t_test,Y_test[0,-1],'ko',label='$Y_T = u(T,X_T)$')
    
    # plt.plot(t_test,Y_pred[1:samples,:].T,'b')
    # plt.plot(t_test,Y_test[1:samples,:].T,'r--')
    # plt.plot(t_test,Y_test[1:samples,-1],'ko')

    # plt.plot([0],Y_test[0,0],'ks',label='$Y_0 = u(0,X_0)$')
    
    plt.xlabel('$t$')
    plt.ylabel('$Y_t = u(t,X_t)$')
    plt.title('100-dimensional Black-Scholes-Barenblatt')
    plt.legend()
    
    plt.savefig('test.png')


if __name__ == "__main__":

    # tf.compat.v1.disable_eager_execution()
    num_threads = 5
    os.environ["OMP_NUM_THREADS"] = "5"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "5"
    os.environ["TF_NUM_INTEROP_THREADS"] = "5"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    tf.config.threading.set_inter_op_parallelism_threads(
        num_threads
    )
    tf.config.threading.set_intra_op_parallelism_threads(
        num_threads
    )
    tf.config.set_soft_device_placement(True)

    args = parser.parse_args()

    test_2()