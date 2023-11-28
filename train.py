import tensorflow as tf
from tensorflow import keras
import numpy as np
import scipy
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib.pyplot as plt
import random


from model import ResNN
from utils.parser import parser


class temporal_difference_learning():
    def __init__(self,T,N,batch_size,d_var,d_hidden,G_func,b_func,sigma_func,f_func,g_func,d_g_func,poisson_lambda,config_name):
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
        self.model = ResNN(d_hidden=d_hidden)

        # optimizer
        # adam optimizer with lr_0 = 5e-5 and decay by 0.2 every 100 iterations
        lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
            [100*self.N,200*self.N,300*self.N,400*self.N], [5e-5,1e-5,2e-6,4e-7,8e-8])
        self.optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    
    def train_loop(self):
        # sample Brownian motion
        self.dW = scipy.stats.norm.rvs(scale=math.sqrt(self.dt),size=(self.batch_size,self.N,self.d_var))
        # sample jumps
        exp_time = scipy.stats.expon.rvs(scale=1/self.poisson_lambda,size=(self.batch_size,1,1))
        while np.min(np.sum(exp_time,axis=1)) < self.T:
            # print(np.min(np.sum(exp_time,axis=1)))
            exp_time = np.concatenate((exp_time,scipy.stats.expon.rvs(scale=1/self.poisson_lambda,size=(self.batch_size,1,1))),axis=1)   # sample exponential as arrival time
        exp_time = (np.cumsum(exp_time,axis=1) // self.dt).astype(int) # discretize continuous jump times to time intervals
        jump_count = np.zeros((self.batch_size,self.N,1))
        jump_count[np.where(exp_time<self.N)[0],exp_time[exp_time<self.N]] = jump_count[np.where(exp_time<self.N)[0],exp_time[exp_time<self.N]]+1 
        # TODO ⬆️we currently ignore the possibility that multiple jumps may happen in the same time interval
        # jump size distribution 
        J = scipy.stats.norm.rvs(loc=0.4,scale=0.25,size=(self.batch_size,self.N,self.d_var)) # normal
        if args.config_name == 'jump_diffusion_d':
            J = np.ones((self.batch_size,self.N,self.d_var)) * 0.1
        J = np.multiply(jump_count,J)
        J = np.concatenate((J,np.zeros((self.batch_size,1,self.d_var))),axis=1)
        # print(sum(J))
        # exit(0)

        for n in range(self.N):
            # eq 2.9
            # dimensions:
            # self.X[:,n,:]: [bs,d]
            # self.sigma_func(self.X[:,n,:]): [bs,d,d]
            # self.dW[:,n,:]: [bs,d,d]
            # self.dW[:,n,:]: [bs,d]
            self.X[:,n+1,:] = self.X[:,n,:] + self.b_func(self.X[:,n,:]) * self.dt + tf.einsum('bij,bj->bi',self.sigma_func(self.X[:,n,:]),self.dW[:,n,:]) + G_func(self.X[:,n,:],J[:,n,:]) - self.dt * int_G_nv_dz_func(self.X[:,n,:],self.poisson_lambda,0.4,0.25)
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

                # compute loss
                # dimensions:
                # self.X[:,n,:]: [bs,d]
                # self.sigma_func(self.X[:,n,:]): [bs,d,d]
                # self.dW[:,n,:]: [bs,d,d]
                # self.dW[:,n,:]: [bs,d]
                TD_error = -self.f_func(self.X[:,n,:],self.poisson_lambda) * self.dt + tf.einsum('bi,bi->b',tf.einsum('bji,bj->bi',self.sigma_func(self.X[:,n,:]),dN1),self.dW[:,n,:]) + (N1_jump-N1) - self.dt * N2 + N1 - N1_next
                # TD_error = -self.f_func(self.X[:,n,:]) * self.dt + tf.einsum('bi,bi->b',tf.einsum('bji,bj->bi',self.sigma_func(self.X[:,n,:]),dN1),self.dW[:,n]) - self.dt * N2 + N1 - N1_next
                Loss_1 = tf.linalg.norm(TD_error,ord=2)**2
                Loss_2 = tf.linalg.norm(N1_T - self.g_func(self.X[:,-1,:]),ord=2)**2/(self.N)
                Loss_3 = tf.linalg.norm(dN1_T - self.d_g_func(self.X[:,-1,:]),ord=2)**2/(self.N)
                Loss_4 = tf.math.abs(tf.math.reduce_sum(N1_jump - N1- self.dt * N2))
                loss_4_weight = 1/self.batch_size
                Loss = Loss_1 + Loss_2 + Loss_3 + loss_4_weight * Loss_4
                self.optimizer.minimize(Loss,self.model.trainable_weights,tape=tape)
                del tape
        print(Loss,Loss_4,self.model(self.t[0],np.ones((1,self.d_var)))[:,0])

            


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

    # set seed
    os.environ['PYTHONHASHSEED']=str(args.seed)
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.config_name == 'pure_brownian':
        from config.pure_brownian import f_func,b_func,G_func,sigma_func,int_G_nv_dz_func,g_func,d_g_func,u_exact
    elif args.config_name == 'pure_jump_1':
        from config.pure_jump import f_func,b_func,G_func,sigma_func,int_G_nv_dz_func,g_func,d_g_func,u_exact
        args.d_var = 1
        args.batch_size = 1000
    elif args.config_name == 'jump_diffusion_1':
        from config.jump_diffusion import f_func,b_func,G_func,sigma_func,int_G_nv_dz_func,g_func,d_g_func,u_exact
        args.d_var = 1
        args.batch_size = 250
    elif args.config_name == 'jump_diffusion_d':
        from config.jump_diffusion_d import f_func,b_func,G_func,sigma_func,int_G_nv_dz_func,g_func,d_g_func,u_exact
        args.d_var = 100
        args.batch_size = 500
        args.d_hidden = args.d_var + 10

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
    # sample jumps
    jump_count = np.zeros((args.batch_size,args.N,1))
    exp_time = scipy.stats.expon.rvs(scale=1/args.poisson_lambda,size=(args.batch_size,1,1))
    while np.min(np.sum(exp_time,axis=1)) < args.T:
        # print(np.min(np.sum(exp_time,axis=1)))
        exp_time = np.concatenate((exp_time,scipy.stats.expon.rvs(scale=1/args.poisson_lambda,size=(args.batch_size,1,1))),axis=1)   # sample exponential as arrival time
    exp_time = (np.cumsum(exp_time,axis=1) // dt).astype(int) # discretize continuous jump times to time intervals
    jump_count = np.zeros((args.batch_size,args.N,1))
    jump_count[np.where(exp_time<args.N)[0],exp_time[exp_time<args.N]] = jump_count[np.where(exp_time<args.N)[0],exp_time[exp_time<args.N]]+1 
    # TODO ⬆️we currently ignore the possibility that multiple jumps may happen in the same time interval
    # jump size distribution
    J = scipy.stats.norm.rvs(loc=0.4,scale=0.25,size=(args.batch_size,args.N,args.d_var)) # normal
    # J = np.ones((args.batch_size,args.N,args.d_var)) * 0.1
    J = np.multiply(jump_count,J)
    J = np.concatenate((J,np.zeros((args.batch_size,1,args.d_var))),axis=1)
    
    X = np.ones((args.batch_size,args.N+1,args.d_var))
    X[:,-1,:] = np.random.randn(args.batch_size,args.d_var)
    for n in range(args.N):
        X[:,n+1,:] = X[:,n,:] + b_func(X[:,n,:]) * dt + tf.einsum('bij,bj->bi',sigma_func(X[:,n,:]),dW[:,n,:]) + G_func(X[:,n,:],J[:,n,:]) - dt * int_G_nv_dz_func(X[:,n,:],args.poisson_lambda,0.4,0.25)
    
    t = tf.range(0,args.T+dt,dt)
    t_test = np.repeat(np.expand_dims(t,0),args.batch_size,axis=0)
    Y_test = np.reshape(u_exact(np.reshape(t_test,[-1,1]),args.T, np.reshape(X,[-1,args.d_var])),[args.batch_size,-1])

    Y_pred = np.zeros((args.batch_size,args.N+1))
    for n in range(args.N+1):
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
    
    plt.savefig(f'figures/{args.config_name}_a_loss4.png')


