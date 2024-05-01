import argparse
import numpy as np
import random
import pandas as pd

from folktables import ACSDataSource, ACSIncome, ACSEmployment, ACSPublicCoverage

from utils_dis import histogram_RR, denoise_histogram_RR, histogram_to_freq 
from utils_dis import duchi_algo, piecewise_algo, hybrid_algo
from a3m_dis import opt_variance, a3m_perturb

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
    parser.add_argument("--runs", help="independent runs", type=int, default=1000) 
    # a3m
    parser.add_argument("--bin_size", help="bin length", type=float, default=0.5)
    parser.add_argument("--axRatio", help="ratio between amax/xmax", type=float, default=4)
    parser.add_argument("--s", help="split ratio", type=float, default=0.1)
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

    error_laplace = np.zeros(epsilon_array.size)
    error_gaussian = np.zeros(epsilon_array.size)
    error_duchi = np.zeros(epsilon_array.size)
    error_piecewise = np.zeros(epsilon_array.size)
    error_hybrid = np.zeros(epsilon_array.size)
    error_a3m_pure = np.zeros(epsilon_array.size)
    error_a3m_app = np.zeros(epsilon_array.size)
    
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

        # split for 3am
        data_1 = data[0:int(split_ratio*n)]
        data_2 = data[int(split_ratio*n):n]
        
        
        for i in range(epsilon_array.size):
            epsilon = epsilon_array[i] * 1.
            for run in range(args.runs):
                """ 
                    laplace 
                """
                laplace_scale = 2 * args.beta / epsilon
                laplace_noise = np.random.laplace(loc=np.zeros(n),scale=laplace_scale)
                laplace_data = data + laplace_noise
                laplace_mean = np.sum(laplace_data) / n
                error_laplace[i] += (true_mean - laplace_mean) ** 2 / args.runs
                """ 
                    duchi takes input from [-1,1]
                """
                duchi_output = duchi_algo(data/args.beta, epsilon)
                duchi_data = args.beta * duchi_output 
                duchi_mean = np.sum(duchi_data) / n
                error_duchi[i] += (true_mean - duchi_mean) ** 2 / args.runs
                """ 
                    piecewise takes input from [-1,1]
                """
                piecewise_output = piecewise_algo(data/args.beta, epsilon)
                piecewise_data = args.beta * piecewise_output 
                piecewise_mean = np.sum(piecewise_data) / n
                error_piecewise[i] += (true_mean - piecewise_mean) ** 2 / args.runs
                """ 
                    hybrid takes input in [-1,1]
                """
                hybrid_outcome = hybrid_algo(data/args.beta, epsilon)
                hybrid_data = args.beta * hybrid_outcome
                hybrid_mean = np.sum(hybrid_data) / n
                error_hybrid[i] += (true_mean - hybrid_mean) ** 2 / args.runs
                """ 
                    a3m 
                """
                # compute noisy histogram with randomize response
                true_histogram_1, noisy_histogram_1 = histogram_RR(data_1, -args.beta, args.beta, args.bin_size, epsilon)    
                true_histogram_2, noisy_histogram_2 = histogram_RR(data_2, -args.beta, args.beta, args.bin_size, epsilon)    
                # convert to frequency
                true_freq = histogram_to_freq(true_histogram_2, -args.beta, args.beta, args.bin_size)
                noisy_freq = histogram_to_freq(noisy_histogram_1, -args.beta, args.beta, args.bin_size)
                # denoise the histogram and convert to frequency
                estimated_freq = denoise_histogram_RR(noisy_histogram_1, -args.beta, args.beta, args.bin_size, epsilon)
                """ pure """
                # use estimated freq to generate a3m noise
                noise_values, opt_distribution_pure = opt_variance(estimated_freq, args.beta, args.bin_size, args.axRatio, epsilon, 0)
                # perturb with a3m
                a3m_noise_pure = a3m_perturb(true_histogram_2, args.beta, args.bin_size, noise_values, opt_distribution_pure)
                M = estimated_freq.size
                x_grid = -args.beta + np.array(range(M)) * args.bin_size
                # print(x_grid)
                clean_dis_mean = np.sum(x_grid * true_freq)
                error_a3m_pure[i] += np.power(clean_dis_mean+np.sum(a3m_noise_pure) / (n-int(split_ratio*n))-true_mean, 2) / args.runs
            print(f'Epsilon: {epsilon_array[i]} finished')
            print(f'Laplace scale:{laplace_scale}, error: {error_laplace[i]}')
            print(f'Duchi\'s error: {error_duchi[i]}')
            print(f'Piecewise error: {error_piecewise[i]}')
            print(f'Hybrid error: {error_hybrid[i]}')
            print(f'Pure-a3m error: {error_a3m_pure[i]}')
    
    print(f'Task={args.task} and bin_size={args.bin_size},averaged over {args.runs} runs')
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
    print(f'Pure-A3M error:')
    for i in range(epsilon_array.size):
        print(epsilon_array[i], error_a3m_pure[i])
    print('\n')
