import argparse
import numpy as np
import random
import pandas as pd

from folktables import ACSDataSource, ACSIncome, ACSEmployment, ACSPublicCoverage

from utils_dis import histogram_RR, denoise_histogram_RR, histogram_to_freq 
from utils_dis import compute_gaussian_sigma, duchi_algo, piecewise_algo, hybrid_algo
from utils_dis import sw

def read_data(acs_data, task, beta):
    if task == 0:
        inc_features, inc_labels, _ = ACSIncome.df_to_numpy(acs_data)
        inc_data = np.vstack([inc_features.T,inc_labels.T]).T
        inc_data = (inc_data - inc_data.min(0)) / inc_data.ptp(0)
        inc_data = 2 * beta * inc_data - beta
        return inc_data
    elif task == 1:
        emp_features, emp_labels, _ = ACSEmployment.df_to_numpy(acs_data)
        emp_data = np.vstack([emp_features.T,emp_labels.T]).T
        emp_data = (emp_data - emp_data.min(0)) / emp_data.ptp(0)
        emp_data = 2 * beta * emp_data - beta
        return emp_data
    else:
        pc_features, pc_labels, _ = ACSPublicCoverage.df_to_numpy(acs_data)
        pc_data = np.vstack([pc_features.T,pc_labels.T]).T
        pc_data = (pc_data - pc_data.min(0)) / pc_data.ptp(0)
        pc_data = 2 * beta * pc_data - beta
        return pc_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Seed
    parser.add_argument("--seed", help="random seed", type=int, default=0)
    # range for DATA
    parser.add_argument("--beta", help="range for data", type=float, default=1)
    # independent runs
    parser.add_argument("--runs", help="independent runs", type=int, default=100) 
    # task
    parser.add_argument("--task", type=int, default=0) # 0 is income, 1 is employment, 2 is public coverage
    # number of sampled dimension
    parser.add_argument("--sample_dim", type=int, default=1) # number of sampled dimensions for each user

    
    args = parser.parse_args()
    print(args)
    
    # fix seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    epsilon_array = np.linspace(1, 4, 4)

    error_sw = np.zeros(epsilon_array.size)

    
    """ 
        read data 
    """
    print('Reading data =====>')
    state_list = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
              'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
              'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
              'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
              'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']

    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person', root_dir="data")
    acs_data = data_source.get_data(states=state_list, download=True)
    data_all = read_data(acs_data, args.task, args.beta)
    print('Finish reading.')
    
    total_samples =  data_all.shape[0] # number of samples
    K = data_all.shape[1] # number of dimension

    split_ratio = args.s # proportion of data for frequency estimation for 3am
    for k in range(K):
        print(f'{k}-th dimension')
        data_k = data_all[:,k] # k-th dimension
        true_mean = np.sum(data_k) / total_samples

        # subsample sample_dim/K fraction for estimation
        n = args.sample_dim * int(total_samples/K)
        data = data_k[np.random.choice(total_samples, n, replace=False)]    
        
        for i in range(epsilon_array.size):
            epsilon = epsilon_array[i] * 1. 
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
            print(f'Epsilon: {epsilon_array[i]} finished')
            print(f'SW error: {error_sw[i]}')
    
    
    print(f'SW error:')
    for i in range(epsilon_array.size):
        print(epsilon_array[i], error_sw[i])
    print('\n')
