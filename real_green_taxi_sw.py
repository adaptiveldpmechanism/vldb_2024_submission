import argparse
import numpy as np
import random
import pandas as pd
from scipy import optimize

from utils_dis import histogram_RR, denoise_histogram_RR, histogram_to_freq 
from utils_dis import compute_gaussian_sigma, duchi_algo, piecewise_algo, hybrid_algo
from utils_dis import sw

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Seed
    parser.add_argument("--seed", help="random seed", type=int, default=0)
    # range for DATA
    parser.add_argument("--beta", help="range for data", type=float, default=1)
    # independent runs
    parser.add_argument("--runs", help="independent runs", type=int, default=100) 
    
    args = parser.parse_args()
    print(args)
    
    # fix seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    epsilon_array = np.linspace(1, 4, 4)

    error_sw = np.zeros(epsilon_array.size)

    
    """ 
        read green taxi data 
    """
    split_ratio = args.s # proportion of data for frequency estimation for 3am
    green_data = pd.read_parquet('taxi/green_tripdata_2018-01.parquet', engine='pyarrow')
    green_data['duration'] = (green_data['lpep_dropoff_datetime'] - green_data['lpep_pickup_datetime']).dt.total_seconds()
    green_data_np = green_data['duration'].to_numpy()
    high = 202989
    low = 0
    clipped_data = green_data_np[green_data_np<=high]
    """ 
        map data to [-beta,beta] 
    """
    a = 2*args.beta / (high-low)
    b = args.beta - 2*args.beta*high / (high-low)
    data = a*clipped_data + b
    n = data.shape[0]
    data_1 = data[0:int(split_ratio*n)]
    data_2 = data[int(split_ratio*n):n]
    true_mean = np.sum(data) / n
    
    for i in range(epsilon_array.size):
        epsilon = epsilon_array[i]
        for run in range(args.runs):
            
            """
                sw
            """
            sw_bins = 1024
            sw_bin_size = 2*args.beta/sw_bins
            sw_outcome = sw(data, -args.beta, args.beta, epsilon, sw_bins, sw_bins)
            sw_centers = np.linspace(-args.beta+sw_bin_size/2, args.beta-sw_bin_size/2, sw_bins)
            sw_mean = np.inner(np.array(sw_outcome), sw_centers) / n
            error_sw[i] += (true_mean - sw_mean) ** 2 / args.runs
        print(f'Epsilon: {epsilon} finished')
        print(f'SW error: {error_sw[i]}')
    
    print(f'On Green Taxi Data with beta={args.beta} and bin_size={args.bin_size},averaged over {args.runs} runs')

    print(f'SW error:')
    for i in range(epsilon_array.size):
        print(epsilon_array[i], error_sw[i])
    print('\n')
    
