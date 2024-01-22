import tensorflow as tf
from tensorflow import keras
import numpy as np
import scipy

# Part I: test random distribution generation
from utils.sample import sample_brownian,sample_jump

class TestSampleingFunctions:

    def test_sample_brownian(self):
        # test output shape
        test_output = sample_brownian(batch_size=30,N=50,d_var=10,dt=0.1)
        assert test_output.shape == (30,50,10)
        # test some statistics (1st - 4th moments are close to theoretical values)
        tolerence = 1e-4
        test_output = sample_brownian(batch_size=10000,N=500,d_var=10,dt=0.1)
        # 1st moment      
        assert np.abs(np.mean(test_output,axis=None)) < tolerence
        # 2nd moment
        assert np.abs(scipy.stats.moment(test_output,moment=2,axis=None)-0.1) < tolerence
        # 3rd moment
        assert np.abs(scipy.stats.moment(test_output,moment=3,axis=None)) < tolerence
        # 4th moment
        assert np.abs(scipy.stats.moment(test_output,moment=4,axis=None)-0.03) < tolerence

    def test_sample_jump(self):
        # test output shape
        test_output,_ = sample_jump(poisson_lambda=0.5,batch_size=30,T=10,N=50,d_var=10,dt=0.2,jump_type='normal')
        assert test_output.shape == (30,51,10)
        # test some statistics (1st - 4th moments of intervals & jump sizes are close to theoretical values)
        tolerence = 1e-4
        test_output,poisson_time = sample_jump(poisson_lambda=0.5,batch_size=10000,T=1000,N=5000,d_var=10,dt=0.2,jump_type='normal')
        # time intervals
        # 1st moment
        assert np.abs(np.mean(poisson_time,axis=None)-2) < 0.01
        # 2nd moment
        assert np.abs(np.std(poisson_time,axis=None)-2) < 0.01
        # 3rd moment
        assert np.abs(scipy.stats.skew(poisson_time,axis=None)-2) < 0.01
        # 4th moment
        assert np.abs(scipy.stats.kurtosis(poisson_time,axis=None)-6) < 0.1

        # take out nonzero jumps
        test_output = test_output[np.nonzero(test_output)]
        # 1st moment
        assert np.abs(np.mean(test_output,axis=None)-0.4) < tolerence
        # 2nd moment
        assert np.abs(scipy.stats.moment(test_output,moment=2,axis=None)-0.25**2) < tolerence
        # 3rd moment
        assert np.abs(scipy.stats.moment(test_output,moment=3,axis=None)) < tolerence
        # 4th moment
        assert np.abs(scipy.stats.moment(test_output,moment=4,axis=None)-3*0.25**4) < tolerence
        
        

# Part II: test option configs
class TestOptionConfig:

    def test_pure_jump_config(self):
        from config.pure_jump import f_func,b_func,G_func,sigma_func,int_G_nv_dz_func,g_func,d_g_func,u_exact
        # test values
        x = np.ones((10,1))
        z = np.zeros((10,1))
        assert np.array_equal(f_func(x,poisson_lambda=0.5), np.zeros((10,)))
        assert np.array_equal(b_func(x), np.zeros((10,1)))
        assert np.array_equal(G_func(x,z), np.zeros((10,1)))
        assert np.array_equal(sigma_func(x), np.zeros((10,1,1)))
        assert np.array_equal(int_G_nv_dz_func(x,1,0,1), np.ones((10,1))*(np.exp(1/2)-1))
        assert np.array_equal(g_func(x), np.ones((10,)))
        assert np.array_equal(d_g_func(x), np.ones((10,1)))
        assert np.array_equal(u_exact(1,1,x), np.ones((10,1)))


    def test_jump_diffusion_1_config(self):
        from config.jump_diffusion import f_func,b_func,G_func,sigma_func,int_G_nv_dz_func,g_func,d_g_func,u_exact
        # test values
        x = np.ones((10,1))
        z = np.zeros((10,1))
        assert np.array_equal(f_func(x,poisson_lambda=0.5), np.zeros((10,)))
        assert np.array_equal(b_func(x), np.zeros((10,1)))
        assert np.array_equal(G_func(x,z), np.zeros((10,1)))
        assert np.array_equal(sigma_func(x), np.repeat(np.expand_dims(np.identity(x.shape[-1],dtype=np.float32) * 0.4,0),10,axis=0))
        assert np.array_equal(int_G_nv_dz_func(x,1,0,1), np.ones((10,1))*(np.exp(1/2)-1))
        assert np.array_equal(g_func(x), np.ones((10,)))
        assert np.array_equal(d_g_func(x), np.ones((10,1)))
        assert np.array_equal(u_exact(1,1,x), np.ones((10,1)))

    def test_jump_diffusion_d_config(self):
        from config.jump_diffusion_d import f_func,b_func,G_func,sigma_func,int_G_nv_dz_func,g_func,d_g_func,u_exact
        # test values
        x = np.ones((10,20))
        z = np.zeros((10,20))
        assert np.array_equal(f_func(x,poisson_lambda=0.5), - np.ones(10) * 0.095)
        assert np.array_equal(b_func(x), np.zeros((10,20)))
        assert np.array_equal(G_func(x,z), np.zeros((10,20)))
        assert np.array_equal(sigma_func(x), np.repeat(np.expand_dims(np.identity(x.shape[-1],dtype=np.float32) * 0.3,0),10,axis=0))
        assert np.array_equal(int_G_nv_dz_func(x,1,0,1), np.ones(x.shape) * 0.1)
        assert np.allclose(g_func(x), np.ones((10,)))
        assert np.allclose(d_g_func(x), np.ones((10,20))*0.1)

    def test_barrier_config(self):
        from config.barrier_option import f_func,b_func,G_func,sigma_func,int_G_nv_dz_func,g_func,d_g_func,u_exact
        # test values
        x = np.ones((10,1))
        z = np.zeros((10,1))
        state = np.ones((10,1))
        assert np.array_equal(f_func(x,poisson_lambda=0.5), np.zeros((10,)))
        assert np.array_equal(b_func(x), np.zeros((10,1)))
        assert np.array_equal(G_func(x,z), np.zeros((10,1)))
        assert np.array_equal(sigma_func(x), np.repeat(np.expand_dims(np.identity(x.shape[-1],dtype=np.float32) * 0.4,0),10,axis=0))
        assert np.array_equal(int_G_nv_dz_func(x,1,0,1), np.ones((10,1))*(np.exp(1/2)-1))
        assert np.array_equal(g_func(x,state), np.ones((10,)))
        assert np.array_equal(d_g_func(x,state), np.ones((10,1)))
        assert np.array_equal(u_exact(1,1,x,state), np.ones((10,1)))


    def test_asian_config(self):
        from config.asian_option import f_func,b_func,G_func,sigma_func,int_G_nv_dz_func,g_func,d_g_func,u_exact
        # test values
        x = np.ones((10,1))
        z = np.zeros((10,1))
        state = np.ones((10,1))
        assert np.array_equal(f_func(x,poisson_lambda=0.5), np.zeros((10,)))
        assert np.array_equal(b_func(x), np.zeros((10,1)))
        assert np.array_equal(G_func(x,z), np.zeros((10,1)))
        assert np.array_equal(sigma_func(x), np.repeat(np.expand_dims(np.identity(x.shape[-1],dtype=np.float32) * 0.4,0),10,axis=0))
        assert np.array_equal(int_G_nv_dz_func(x,1,0,1), np.ones((10,1))*(np.exp(1/2)-1))
        assert np.array_equal(g_func(x,state), np.ones((10,)))
        assert np.array_equal(d_g_func(x,state,1), np.ones((10,1)))
        assert np.array_equal(u_exact(1,1,x,state), np.ones((10,1)))


# Part III: test loss functions (mathematically correct and their functionalities)
class TestLossFunctions:
    
    def test_loss_1(self):
        from config.pure_jump import f_func,sigma_func
        poisson_lambda = 0.3
        dt = 0.1
        X_n = np.ones((10,1))
        dW_n = np.ones((10,1))
        X_next = X_n + dt * dW_n
        # true model
        N1 = X_n
        N1_next = X_next
        dN1 = np.ones((10,1))
        N1_jump = N1
        N2 = np.zeros((10,1))

        TD_error = -f_func(X_n,poisson_lambda) * dt + tf.einsum('bi,bi->b',tf.einsum('bji,bj->bi',sigma_func(X_n),dN1),dW_n) + (N1_jump-N1) - dt * N2 + N1 - N1_next
        Loss_1 = tf.linalg.norm(TD_error,ord=2)**2
        assert np.abs(Loss_1 - 1) <= 1e-6

    def test_loss_2(self):
        from config.pure_jump import g_func
        X_T = N1_T = np.random.rand(10,10)
        Loss_2 = tf.linalg.norm(N1_T - g_func(X_T),ord=2)**2/10
        assert Loss_2 == 0

    def test_loss_3(self):
        from config.pure_jump import d_g_func
        X_T = np.random.rand(10,10)
        dN1_T = np.ones((10,10))
        Loss_3 = tf.linalg.norm(dN1_T - d_g_func(X_T),ord=2)**2/10
        assert Loss_3 == 0

    def test_loss_4(self):
        dt = 0.01
        X_n = np.ones((10,1))
        # true model
        N1 = X_n
        N1_jump = N1
        N2 = np.zeros((10,1))
        Loss_4 = tf.math.abs(tf.math.reduce_sum(N1_jump - N1- dt * N2))
        assert Loss_4 == 0

        

# Part VI: test the whole method (compare with the true price)
        