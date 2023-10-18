import argparse

parser = argparse.ArgumentParser()

# data
parser.add_argument('--d_var',default=1,type=int,help='input dimension of x')

# process
parser.add_argument('--poisson_lambda',type=float,default=1, help='lambda for poisson distribution in jump generation')
parser.add_argument('--jump_dist',choices=['normal','exp','uniform','bernoulli'],default='normal',
                    help='distribution for jump size')
parser.add_argument('--config_name',choices=['pure_jump','pure_brownian','jump_diffusion'])

# training
parser.add_argument('--batch_size',type=int,default=50,help='number of considered path in each training step (M in the original paper)')
parser.add_argument('--T',type=float,default=1,help='maturity time')
parser.add_argument('--N',type=int,default=50,help='number of equally divided intervals in [0,T]')

# model
parser.add_argument('--d_hidden',type=int,default=25,help='hidden dimension in neural netwrork')
