import tensorflow as tf
from tensorflow import keras
import numpy as np
import scipy
import math
import os
import matplotlib.pyplot as plt
import random


from model import ResNN, ResNN_state
from utils.parser import parser
from utils.sample import sample_brownian


class temporal_difference_learning():
    def __init__(self,T:int,N:int,batch_size:int,d_var:int,d_hidden:int,
                 G_func,b_func,sigma_func,f_func,
                 g_func,d_g_func,poisson_lambda:float,config_name:str):
        """
        T: expiration time of an option
        N: number of stocks
        batch_size
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
        self.config_name = config_name
        # generate random sample
        self.dt = T/N
        self.t = tf.range(0,T+self.dt,self.dt)
        self.X = np.ones((args.batch_size,args.N+1,args.d_var))
        self.X[:,-1,:] = np.random.randn(args.batch_size,args.d_var)
        self.poisson_lambda = poisson_lambda

        # functional component
        self.b_func = b_func
        self.sigma_func = sigma_func
        self.G_func = G_func
        self.f_func = f_func
        self.g_func = g_func
        self.d_g_func = d_g_func

        # neural network
        if self.config_name in ['asian','barrier']:
            self.model = ResNN_state(d_hidden=d_hidden)
        else:
            self.model = ResNN(d_hidden=d_hidden)

        # optimizer
        # adam optimizer with lr_0 = 5e-5 and decay by 0.2 every 100 iterations
        lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
            [100*self.N,200*self.N,300*self.N,400*self.N], [5e-5,1e-5,2e-6,4e-7,8e-8])
        self.optimizer = keras.optimizers.legacy.Adam(learning_rate=lr_schedule)
    
    def train_loop(self):
        # sample Brownian motion
        self.dW = sample_brownian(batch_size=self.batch_size,N=self.N,d_var=self.d_var,dt=self.dt)

        # state vector to record
        # (1) past average price, for Asian options
        # (2) flag to represent whether we are within the boundary, for barrier options
        self.state = np.ones((self.batch_size,self.N+1))

        for n in range(self.N):
            # eq 2.9
            # dimensions:
            # self.X[:,n,:]: [bs,d]
            # self.sigma_func(self.X[:,n,:]): [bs,d,d]
            # self.dW[:,n,:]: [bs,d,d]
            # self.dW[:,n,:]: [bs,d]
            self.X[:,n+1,:] = self.X[:,n,:] + self.b_func(self.X[:,n,:]) * self.dt + tf.einsum('bij,bj->bi',self.sigma_func(self.X[:,n,:]),self.dW[:,n,:])

            # Asian options: update the average price (for illustration we use geometric average here, as it has an analytical BS solution)
            if self.config_name == 'asian':
                self.X = self.X.clip(min=0.01)
                self.state[:,n+1] = np.squeeze(np.power(np.multiply(np.power(np.expand_dims(self.state[:,n],1),n+1), self.X[:,n+1,:]),1/(n+2))) # geometric average
                self.state[:,-1] = np.squeeze(np.power(np.multiply(np.power(np.expand_dims(self.state[:,n+1],1),n+2), self.X[:,-1,:]),1/(n+3)))
            # Barrier options: update if the stock price triggers the barrier condition
            elif self.config_name == 'barrier':
                self.state[:,n+1] = np.squeeze(barrier_fn(np.expand_dims(self.state[:,n],1),self.X[:,n+1,:]))
                self.state[:,-1] = np.squeeze(barrier_fn(np.expand_dims(self.state[:,n],1),self.X[:,-1,:]))
            
            with tf.GradientTape(persistent=True) as tape:
                if self.config_name in ['asian','barrier']:
                    # X_n
                    X_n = tf.Variable(self.X[:,n,:],dtype=tf.float32)
                    output = self.model(self.t[n],np.expand_dims(self.state[:,n],1),X_n) # output shape [bs,2]
                    N1 = output[:,0]; N2 = output[:,1] # [bs,]
                    dN1 = tape.gradient(N1,X_n) # [bs,d]
                    # X_T
                    X_T = tf.Variable(self.X[:,-1,:],dtype=tf.float32)
                    N1_T = self.model(self.t[-1],np.expand_dims(self.state[:,-1],1),X_T)[:,0] 
                    dN1_T = tape.gradient(N1_T,X_T) # [bs,d]
                    # X_n+1
                    X_next = tf.Variable(self.X[:,n+1,:],dtype=tf.float32)
                    N1_next = self.model(self.t[n+1],np.expand_dims(self.state[:,n+1],1),X_next)[:,0]
                
                else:
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
                

                # compute loss
                # dimensions:
                # self.X[:,n,:]: [bs,d]
                # self.sigma_func(self.X[:,n,:]): [bs,d,d]
                # self.dW[:,n,:]: [bs,d,d]
                # self.dW[:,n,:]: [bs,d]
                TD_error = -self.f_func(self.X[:,n,:],self.poisson_lambda) * self.dt + tf.einsum('bi,bi->b',tf.einsum('bji,bj->bi',self.sigma_func(self.X[:,n,:]),dN1),self.dW[:,n,:])  + N1 - N1_next
                Loss_1 = tf.linalg.norm(TD_error,ord=2)**2
                if self.config_name == 'asian':
                    Loss_2 = tf.linalg.norm(N1_T - self.g_func(self.X[:,-1,:],np.expand_dims(self.state[:,-1],1)),ord=2)**2/(self.N)
                    Loss_3 = tf.linalg.norm(dN1_T - self.d_g_func(self.X[:,-1,:],np.expand_dims(self.state[:,-1],1),self.N),ord=2)**2/(self.N)
                elif self.config_name == 'barrier':
                    Loss_2 = tf.linalg.norm(N1_T - self.g_func(self.X[:,-1,:],np.expand_dims(self.state[:,-1],1)),ord=2)**2/(self.N)
                    Loss_3 = tf.linalg.norm(dN1_T - self.d_g_func(self.X[:,-1,:],np.expand_dims(self.state[:,-1],1)),ord=2)**2/(self.N)
                else:
                    Loss_2 = tf.linalg.norm(N1_T - self.g_func(self.X[:,-1,:]),ord=2)**2/(self.N)
                    Loss_3 = tf.linalg.norm(dN1_T - self.d_g_func(self.X[:,-1,:]),ord=2)**2/(self.N)
                Loss = Loss_1 + Loss_2 + Loss_3
            self.optimizer.minimize(Loss,self.model.trainable_weights,tape=tape)
            # del tape
        if self.config_name in ['barrier','asian']:
            print(Loss_1,Loss_2,Loss_3,self.model(self.t[0],np.ones((1,1)),np.ones((1,self.d_var)))[:,0])
        else:
            print(Loss_1,Loss_2,Loss_3,self.model(self.t[0],np.ones((1,self.d_var)))[:,0])

            


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

    # set seed
    os.environ['PYTHONHASHSEED']=str(args.seed)
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.config_name == 'pure_brownian':
        from config.pure_brownian import f_func,b_func,G_func,sigma_func,int_G_nv_dz_func,g_func,d_g_func,u_exact
    elif args.config_name == 'barrier':
        from config.barrier_option import f_func,b_func,G_func,sigma_func,int_G_nv_dz_func,g_func,d_g_func,u_exact,barrier_fn
        args.d_var = 1
        args.batch_size = 1000
    elif args.config_name == 'asian':
        from config.asian_option import f_func,b_func,G_func,sigma_func,int_G_nv_dz_func,g_func,d_g_func,u_exact
        args.d_var = 1
        args.batch_size = 1000

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
                                           d_g_func=d_g_func,
                                           config_name=args.config_name)
    for i in range(400):
        print(i)
        Trainer.train_loop()
    
    model = Trainer.model

    # test data
    dt = args.T/args.N
    dW = scipy.stats.norm.rvs(scale=math.sqrt(dt),size=(args.batch_size,args.N,args.d_var))
    
    X = np.ones((args.batch_size,args.N+1,args.d_var))
    X[:,-1,:] = np.random.randn(args.batch_size,args.d_var)
    state = np.ones((args.batch_size,args.N+1))
    for n in range(args.N):
        X[:,n+1,:] = X[:,n,:] + b_func(X[:,n,:]) * dt + tf.einsum('bij,bj->bi',sigma_func(X[:,n,:]),dW[:,n,:]) 
        # Asian options: update the average price (for illustration we use geometric average here, as it has an analytical BS solution)
        if args.config_name == 'asian':
            state[:,n+1] = np.squeeze(np.power(np.multiply(np.power(np.expand_dims(state[:,n],1),n+1), X[:,n+1,:]),1/(n+2))) # geometric average
        # Barrier options: update if the stock price triggers the barrier condition
        elif args.config_name == 'barrier':
            state[:,n+1] = np.squeeze(barrier_fn(np.expand_dims(state[:,n],1),X[:,n+1,:]))
    
    t = tf.range(0,args.T+dt,dt)
    t_test = np.repeat(np.expand_dims(t,0),args.batch_size,axis=0)
    if args.config_name == 'asian':
        Y_test = np.reshape(u_exact(np.reshape(t_test,[-1,1]),args.T, np.reshape(X,[-1,args.d_var]),np.reshape(state,[-1,1])),[args.batch_size,-1])
    elif args.config_name == 'barrier':
        Y_test = np.reshape(u_exact(np.reshape(t_test,[-1,1]),args.T, np.reshape(X,[-1,args.d_var]),np.reshape(state,[-1,1])),[args.batch_size,-1])
    else:
        Y_test = np.reshape(u_exact(np.reshape(t_test,[-1,1]),args.T, np.reshape(X,[-1,args.d_var])),[args.batch_size,-1])

    Y_pred = np.zeros((args.batch_size,args.N+1))
    for n in range(args.N+1):
        if args.config_name in ['asian','barrier']:
            output = model(t[n],np.expand_dims(state[:,n],1),X[:,n,:])
        else:
            output = model(t[n],X[:,n,:])
        Y_pred[:,n] = output[:,0].numpy()

    samples = 10
    
    plt.figure()
    print(t_test.shape,Y_test.shape)
    plt.plot(t_test[0,:].T,Y_pred[0,:].T,'b',label='Learned $u(t,X_t)$')
    plt.plot(t_test[0,:].T,Y_test[0,:].T,'r--',label='Exact $u(t,X_t)$')
    plt.plot(t_test[0,-1],Y_test[0,-1],'ko',label='$Y_T = u(T,X_T)$')
    
    plt.plot(t_test[1:samples,:].T,Y_pred[1:samples,:].T,'b')
    plt.plot(t_test[1:samples,:].T,Y_test[1:samples,:].T,'r--')
    plt.plot(t_test[1:samples,-1],Y_test[1:samples,-1],'ko')

    plt.plot([0],Y_test[0,0],'ks',label='$Y_0 = u(0,X_0)$')
    
    plt.xlabel('$t$')
    plt.ylabel('$Y_t = u(t,X_t)$')
    plt.legend()
    
    plt.savefig(f'figures/nojump_{args.config_name}.png')


