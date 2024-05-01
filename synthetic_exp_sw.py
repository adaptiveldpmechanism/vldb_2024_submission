import argparse
import numpy as np
import random
import time

from synthetic_generate import generate_synthetic_data
from utils_dis import sw

"""
Example:
    python synthetic_exp_sw.py --data_type=GAUSSIAN --n=10000 --seed=2
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Seed
    parser.add_argument("--seed", help="random seed", type=int, default=0)
    # Synthetic DATA
    parser.add_argument("--data_type", help="which data to use", type=str, default="GAUSSIAN")
    parser.add_argument("--n", help="overall number of data points", type=int, default=10000)
    parser.add_argument("--low", help="lower limit for clipping", type=float, default=-5)
    parser.add_argument("--high", help="upper limit for clipping", type=float, default=5)
    # range for DATA
    parser.add_argument("--beta", help="range for data", type=float, default=1)
    # Privacy
    # parser.add_argument("--epsilon", help="privacy constraint", type=float, default=1)
    parser.add_argument("--delta", help="privacy constraint", type=float, default=0.00001)
    # independent runs
    parser.add_argument("--runs", help="independent runs", type=int, default=100) 
    parser.add_argument("--s", help="split ratio", type=float, default=0.1)

    
    args = parser.parse_args()
    print(args)
    
    # fix seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    epsilon_array = np.linspace(0.5, 4, 8)
    print('epsilon_array', epsilon_array)
    
    error_sw = np.zeros(epsilon_array.size)
    
    for i in range(epsilon_array.size):
        epsilon = epsilon_array[i]
        for run in range(args.runs):
            """ 
                generate data in [-beta,beta] 
            """
            split_ratio = args.s # proportion of data for frequency estimation for 3am
            data = generate_synthetic_data(args.data_type, args.n, args.low, args.high, args.beta)
            data_1 = data[0:int(split_ratio*args.n)]
            data_2 = data[int(split_ratio*args.n):args.n]
            true_mean = np.sum(data) / args.n
            """
                sw
            """
            sw_bins = 64
            sw_bin_size = 2*args.beta/sw_bins
            sw_outcome = sw(data, -args.beta, args.beta, epsilon, sw_bins, sw_bins)
            sw_centers = np.linspace(-args.beta+sw_bin_size/2, args.beta-sw_bin_size/2, sw_bins)
            sw_mean = np.inner(np.array(sw_outcome), sw_centers) / args.n
            error_sw[i] += (true_mean - sw_mean) ** 2 / args.runs
            
        print(f'Epsilon: {epsilon} finished')
        print(f'SW error: {error_sw[i]}')
    
    print(f'On {args.data_type} with {args.n} data points of low={args.low} and high={args.high} and beta={args.beta},averaged over {args.runs} runs')
    print(f'SW error:')
    for i in range(epsilon_array.size):
        print(epsilon_array[i], error_sw[i])
    print('\n')
            
    
    
