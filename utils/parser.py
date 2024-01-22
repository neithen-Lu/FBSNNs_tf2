import argparse

parser = argparse.ArgumentParser()

# data
parser.add_argument('--d_var',default=1,type=int,help='input dimension of x')

# process
parser.add_argument('--poisson_lambda',type=float,default=0.3, help='lambda for poisson distribution in jump generation')
parser.add_argument('--jump_dist',choices=['normal','exp','uniform','bernoulli'],default='normal',
                    help='distribution for jump size')
parser.add_argument('--config_name',choices=['pure_jump_1','pure_brownian','jump_diffusion_1','jump_diffusion_d','barrier','asian'],
                    help='''pure_brownian: brownian motion
                            pure_jump_1: one dimensional pure jump process
                            jump_diffusion_1: one dimensional jump diffusion process
                            pure_jump_d: multi-dimensional pure jump process
                            jump_diffusion_d: multi-dimensional jump diffusion process
                            ''')

# training
parser.add_argument('--batch_size',type=int,default=50,help='number of considered path in each training step (M in the original paper)')
parser.add_argument('--T',type=float,default=1,help='maturity time')
parser.add_argument('--N',type=int,default=50,help='number of equally divided intervals in [0,T]')
parser.add_argument('--seed',type=int,default=2023,help='random seed')


# model
parser.add_argument('--d_hidden',type=int,default=25,help='hidden dimension in neural netwrork')
