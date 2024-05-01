import argparse
import numpy as np
import random
import time

from synthetic_generate import generate_synthetic_data
from utils_dis import histogram_RR, denoise_histogram_RR, histogram_to_freq 
from utils_dis import duchi_algo, piecewise_algo, hybrid_algo
from a3m_dis import opt_variance, a3m_perturb

"""
Example:
    python synthetic_exp_dis.py --data_type=GAUSSIAN --n=10000
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
    # independent runs
    parser.add_argument("--runs", help="independent runs", type=int, default=1000) 
    # a3m
    parser.add_argument("--bin_size", help="bin length", type=float, default=0.5)
    parser.add_argument("--axRatio", help="ratio between amax/xmax", type=float, default=4)
    parser.add_argument("--s", help="split ratio", type=float, default=0.1)

    
    args = parser.parse_args()
    print(args)
    
    # fix seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    epsilon_array = np.linspace(0.5, 4, 8)

    print('epsilon_array', epsilon_array)

    error_laplace = np.zeros(epsilon_array.size)
    error_duchi = np.zeros(epsilon_array.size)
    error_piecewise = np.zeros(epsilon_array.size)
    error_hybrid = np.zeros(epsilon_array.size)
    error_a3m_pure = np.zeros(epsilon_array.size)
    
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
            # print(data)
            """ 
                laplace 
            """
            laplace_scale = 2 * args.beta / epsilon
            laplace_noise = np.random.laplace(loc=np.zeros(args.n),scale=laplace_scale)
            laplace_data = data + laplace_noise
            laplace_mean = np.sum(laplace_data) / args.n
            error_laplace[i] += (true_mean - laplace_mean) ** 2 / args.runs   
            """ 
                duchi takes input from [-1,1]
            """
            duchi_output = duchi_algo(data/args.beta, epsilon)
            duchi_data = args.beta * duchi_output 
            duchi_mean = np.sum(duchi_data) / args.n
            error_duchi[i] += (true_mean - duchi_mean) ** 2 / args.runs
            """ 
                piecewise takes input from [-1,1]
            """
            piecewise_output = piecewise_algo(data/args.beta, epsilon)
            piecewise_data = args.beta * piecewise_output 
            piecewise_mean = np.sum(piecewise_data) / args.n
            error_piecewise[i] += (true_mean - piecewise_mean) ** 2 / args.runs
            """ 
                hybrid takes input in [-1,1]
            """
            hybrid_outcome = hybrid_algo(data/args.beta, epsilon)
            hybrid_data = args.beta * hybrid_outcome
            hybrid_mean = np.sum(hybrid_data) / args.n
            error_hybrid[i] += (true_mean - hybrid_mean) ** 2 / args.runs
            """ 
                a3m pure and app
            """
            # compute noisy histogram with randomize response
            true_histogram_1, noisy_histogram_1 = histogram_RR(data_1, -args.beta, args.beta, args.bin_size, epsilon)
            # print('true 1:', true_histogram_1)
            # print('noisy 1:',noisy_histogram_1)    
            true_histogram_2, noisy_histogram_2 = histogram_RR(data_2, -args.beta, args.beta, args.bin_size, epsilon)    
            # convert to frequency
            true_freq = histogram_to_freq(true_histogram_2, -args.beta, args.beta, args.bin_size)
            # print(true_freq)
            noisy_freq = histogram_to_freq(noisy_histogram_1, -args.beta, args.beta, args.bin_size)
            # denoise the histogram and convert to frequency
            estimated_freq = denoise_histogram_RR(noisy_histogram_1, -args.beta, args.beta, args.bin_size, epsilon)
            # use estimated freq to generate a3m noise
            noise_values, opt_distribution_pure = opt_variance(estimated_freq, args.beta, args.bin_size, args.axRatio, epsilon, 0)
            # perturb with a3m
            a3m_noise_pure = a3m_perturb(true_histogram_2, args.beta, args.bin_size, noise_values, opt_distribution_pure)
            M = estimated_freq.size
            x_grid = -args.beta + np.array(range(M)) * args.bin_size
            # print(x_grid)
            clean_dis_mean = np.sum(x_grid * true_freq)
            error_a3m_pure[i] += np.power(clean_dis_mean+np.sum(a3m_noise_pure) / (args.n-int(split_ratio*args.n))-true_mean, 2) / args.runs
            
        print(f'Epsilon: {epsilon} finished')
        print(f'Laplace scale:{laplace_scale}, error: {error_laplace[i]}')
        print(f'Duchi\'s error: {error_duchi[i]}')
        print(f'Piecewise error: {error_piecewise[i]}')
        print(f'Hybrid error: {error_hybrid[i]}')
        print(f'Pure-a3m error: {error_a3m_pure[i]}')
    
    print(f'On {args.data_type} with {args.n} data points of low={args.low} and high={args.high} and beta={args.beta} and bin_size={args.bin_size},averaged over {args.runs} runs')
    print(f'Laplace error:')
    for i in range(epsilon_array.size):
        print(epsilon_array[i], error_laplace[i])
    print('\n')
    print(f'Duchi error:')
    for i in range(epsilon_array.size):
        print(epsilon_array[i], error_duchi[i])
    print('\n')
    print(f'Piecewise error:')
    for i in range(epsilon_array.size):
        print(epsilon_array[i], error_piecewise[i])
    print('\n')
    print(f'Hybrid error:')
    for i in range(epsilon_array.size):
        print(epsilon_array[i], error_hybrid[i])
    print('\n')
    print(f'A3M error:')
    for i in range(epsilon_array.size):
        print(epsilon_array[i], error_a3m_pure[i])
    print('\n')
   

            
    
    
