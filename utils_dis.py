import numpy as np
import scipy
from scipy.special import erfc
from scipy import optimize
from numpy import linalg as LA

"""
Input: histogram as a dict
Output: corresponding frequency
"""
def histogram_to_freq(histogram, low, high, bin_size):
    freq = np.zeros(int((high-low) / bin_size)+1)
    for i in histogram:
        freq[i] = histogram[i]
    freq = freq / np.sum(freq)
    return freq
        

"""
Input: true_response and response candidates 
Output: noisy response 
Achieving epsilon-DP
""" 
def general_RR(true_response, response_candidates, epsilon, num_test=1):
    if true_response not in response_candidates:
        print('true response is', true_response)
        print('response candidates are', response_candidates)
        raise ValueError()
    else:
        index_true_response = np.where(response_candidates==true_response)
        weights = np.ones(len(response_candidates))
        weights[index_true_response] = np.exp(epsilon)
        response = np.random.choice(response_candidates, num_test, True, weights/np.sum(weights))
        # his = np.histogram(response, bins=len(response_candidates))
        # print(his)
        return response
        
def histogram_RR(data, low, high, bin_size, epsilon):
    def map_to_bin(item, low, high, bin_size):
        left_bin_index = int(np.floor((item-low) / bin_size))
        left_bin = left_bin_index * bin_size + low
        right_prob = (item-left_bin) / bin_size
        return left_bin_index + np.random.binomial(1, right_prob)
    # map every data item to bin
    bin_indices = [map_to_bin(item, low, high, bin_size) for item in data]
    unique, counts = np.unique(bin_indices, return_counts=True)
    # ground truth
    true_dict = dict(zip(unique, counts))
    # RR 
    num_bins = int((high-low) / bin_size) + 1
    bin_candidates = np.array(range(num_bins)) 
    overall_responses = []
    for i in true_dict:
        count = true_dict[i]
        # RR for every bin
        responses = general_RR(i, bin_candidates, epsilon, count)
        overall_responses.append(responses)
    # combine the results
    noisy_bin_indices = np.concatenate(overall_responses).ravel()
    noisy_unique, noisy_counts = np.unique(noisy_bin_indices, return_counts=True)
    # noisy histogram
    noisy_dict = dict(zip(noisy_unique, noisy_counts))
    return true_dict, noisy_dict
        

def denoise_histogram_RR(histogram_to_denoise, low, high, bin_size, epsilon):
    num_bins = int((high-low) / bin_size) + 1
    # Ax = b, A is the conversion matrix for RR, b is the noisy result
    A = np.zeros((num_bins,num_bins))
    b = np.zeros(num_bins)
    for j in range(num_bins):
        weights = np.ones(num_bins)
        weights[j] = np.exp(epsilon)
        weights = weights/np.sum(weights)
        A[j] = weights
        if j in histogram_to_denoise:
            b[j] = histogram_to_denoise[j] 
        else:
            b[j] = 0
    # print('before denoising:', b)
    # solve
    res = np.linalg.solve(A, b)
    # print('after denoising:', res)
    # post-process the negative counts
    clipped_result = np.clip(res, 0, None)
    # convert to frequency
    clipped_result = clipped_result / np.sum(clipped_result)
    # print('after post-processing:', clipped_result)
    return clipped_result

def compute_gaussian_sigma(beta, epsilon, delta, step_start, step_end, step_chi, prec):
    c = 2 * beta
    found = 0
    for step in range(step_start, step_end):
        chi = step*step_chi
        cur_delta = (erfc(chi) - np.exp(epsilon)*erfc(np.sqrt(chi*chi+epsilon))) / 2
        rel_err = np.abs((cur_delta-delta) / delta)
        sigma = c / np.sqrt(2) / (np.sqrt(chi*chi+epsilon)-chi)
        if np.abs(rel_err) <= prec:
            found = 1
            break
    return sigma, found

def duchi_algo(data, epsilon):
    # set up the bernoulli
    p = (np.exp(epsilon)-1)/(2*np.exp(epsilon)+2) * data + 0.5
    # sample u \in {0,1}
    ber = np.random.binomial(1, p, size=None)
    # map 1 to (exp(eps)+1) / (exp(eps)-1) and 0 to -(exp(eps)+1) / (exp(eps)-1)
    a = 2 * (np.exp(epsilon)+1) / (np.exp(epsilon)-1)
    b = -(np.exp(epsilon)+1) / (np.exp(epsilon)-1)
    return a * ber + b

def piecewise_algo(data, epsilon):
    # set up the piecewise mechanism
    C = (np.exp(epsilon/2) + 1) / (np.exp(epsilon/2) - 1)
    l_data = (C+1)/2 * data - (C-1)/2
    r_data = l_data + C - 1

    """ if data < exp(eps/2) / (exp(eps/2)+1) """
    p_1 = np.exp(epsilon/2) / (np.exp(epsilon/2)+1) * np.ones(data.size)
    ber_1 = np.random.binomial(1, p_1, size=None)
    # sample uniformly from [l_data, r_data]
    t_1 = np.random.uniform(low=l_data,high=r_data,size=data.size)
    
    """ else: sample uniformly from [-C,l_data) U (r_data,C] """
    # flip coin with head prob (l_data + C) / (C + 1)
    p_2 = (l_data+C) / (C+1)
    ber_2 = np.random.binomial(1, p_2, size=None)
    # if heads, uniform from [-C, l_data]
    t_2 = np.random.uniform(low=-C,high=l_data,size=data.size)
    # else uniform sample from [r_data, C]
    t_3 = np.random.uniform(low=r_data,high=C,size=data.size)
    
    """ combine results """
    t = np.multiply(ber_1,t_1) + np.multiply(1-ber_1,np.multiply(ber_2,t_2)+np.multiply(1-ber_2,t_3))
    return t

def hybrid_algo(data, epsilon):
    epsilon_star = -np.log(27) + np.log(-5+2*np.power(6353-405*np.power(241,0.5),1./3)+\
                                          2*np.power(6353+405*np.power(241,0.5),1./3))
    alpha = 0
    if epsilon > epsilon_star:
        alpha = 1 - np.exp(-epsilon/2)
    p_piecewise = np.ones(data.size) * alpha
    ber_piecewise = np.random.binomial(1, p_piecewise, size=None)
    return np.multiply(ber_piecewise,piecewise_algo(data,epsilon)) + np.multiply(1-ber_piecewise,duchi_algo(data,epsilon))
    
def sw(ori_samples, l, h, eps, randomized_bins=1024, domain_bins=1024):
    ee = np.exp(eps)
    w = ((eps * ee) - ee + 1) / (2 * ee * (ee - 1 - eps)) * 2
    p = ee / (w * ee + 1)
    q = 1 / (w * ee + 1)

    samples = (ori_samples - l) / (h - l)
    randoms = np.random.uniform(0, 1, len(samples))

    noisy_samples = np.zeros_like(samples)

    # report
    index = randoms <= (q * samples)
    noisy_samples[index] = randoms[index] / q - w / 2
    index = randoms > (q * samples)
    noisy_samples[index] = (randoms[index] - q * samples[index]) / p + samples[index] - w / 2
    index = randoms > q * samples + p * w
    noisy_samples[index] = (randoms[index] - q * samples[index] - p * w) / q + samples[index] + w / 2

    # report matrix
    m = randomized_bins
    n = domain_bins
    m_cell = (1 + w) / m
    n_cell = 1 / n

    transform = np.ones((m, n)) * q * m_cell
    for i in range(n):
        left_most_v = (i * n_cell)
        right_most_v = ((i + 1) * n_cell)

        ll_bound = int(left_most_v / m_cell)
        lr_bound = int((left_most_v + w) / m_cell)
        rl_bound = int(right_most_v / m_cell)
        rr_bound = int((right_most_v + w) / m_cell)

        ll_v = left_most_v - w / 2
        rl_v = right_most_v - w / 2
        l_p = ((ll_bound + 1) * m_cell - w / 2 - ll_v) * (p - q) + q * m_cell
        r_p = ((rl_bound + 1) * m_cell - w / 2 - rl_v) * (p - q) + q * m_cell
        if rl_bound > ll_bound:
            transform[ll_bound, i] = (l_p - q * m_cell) * ((ll_bound + 1) * m_cell - w / 2 - ll_v) / n_cell * 0.5 + q * m_cell
            transform[ll_bound + 1, i] = p * m_cell - (p * m_cell - r_p) * (rl_v - ((ll_bound + 1) * m_cell - w / 2)) / n_cell * 0.5
        else:
            transform[ll_bound, i] = (l_p + r_p) / 2
            transform[ll_bound + 1, i] = p * m_cell

        lr_v = left_most_v + w / 2
        rr_v = right_most_v + w / 2
        r_p = (rr_v - (rr_bound * m_cell - w / 2)) * (p - q) + q * m_cell
        l_p = (lr_v - (lr_bound * m_cell - w / 2)) * (p - q) + q * m_cell
        if rr_bound > lr_bound:
            if rr_bound < m:
                transform[rr_bound, i] = (r_p - q * m_cell) * (rr_v - (rr_bound * m_cell - w / 2)) / n_cell * 0.5 + q * m_cell

            transform[rr_bound - 1, i] = p * m_cell - (p * m_cell - l_p) * ((rr_bound * m_cell - w / 2) - lr_v) / n_cell * 0.5
			
        else:
            transform[rr_bound, i] = (l_p + r_p) / 2
            transform[rr_bound - 1, i] = p * m_cell

        if rr_bound - 1 > ll_bound + 2:
            transform[ll_bound + 2: rr_bound - 1, i] = p * m_cell

    max_iteration = 10000
    loglikelihood_threshold = 1e-3
    ns_hist, _ = np.histogram(noisy_samples, bins=randomized_bins, range=(-w / 2, 1 + w / 2))
    return EM(n, ns_hist, transform, max_iteration, loglikelihood_threshold) * len(ori_samples)
    # return EMS(n, ns_hist, transform, max_iteration, loglikelihood_threshold) * len(ori_samples)


def EMS(n, ns_hist, transform, max_iteration, loglikelihood_threshold):
    # smoothing matrix
    smoothing_factor = 2
    binomial_tmp = [scipy.special.binom(smoothing_factor, k) for k in range(smoothing_factor + 1)]
    smoothing_matrix = np.zeros((n, n))
    central_idx = int(len(binomial_tmp) / 2)
    for i in range(int(smoothing_factor / 2)):
        smoothing_matrix[i, : central_idx + i + 1] = binomial_tmp[central_idx - i:]
    for i in range(int(smoothing_factor / 2), n - int(smoothing_factor / 2)):
        smoothing_matrix[i, i - central_idx: i + central_idx + 1] = binomial_tmp
    for i in range(n - int(smoothing_factor / 2), n):
        remain = n - i - 1
        smoothing_matrix[i, i - central_idx + 1:] = binomial_tmp[: central_idx + remain]
    row_sum = np.sum(smoothing_matrix, axis=1)
    smoothing_matrix = (smoothing_matrix.T / row_sum).T

    # EMS
    theta = np.ones(n) / float(n)
    theta_old = np.zeros(n)
    r = 0
    sample_size = sum(ns_hist)
    old_logliklihood = 0

    while LA.norm(theta_old - theta, ord=1) > 1 / sample_size and r < max_iteration:
        theta_old = np.copy(theta)
        X_condition = np.matmul(transform, theta_old)

        TMP = transform.T / X_condition

        P = np.copy(np.matmul(TMP, ns_hist))
        P = P * theta_old

        theta = np.copy(P / sum(P))

        # Smoothing step
        theta = np.matmul(smoothing_matrix, theta)
        theta = theta / sum(theta)

        logliklihood = np.inner(ns_hist, np.log(np.matmul(transform, theta)))
        imporve = logliklihood - old_logliklihood

        if r > 1 and abs(imporve) < loglikelihood_threshold:
            # print("stop when", imporve / old_logliklihood, loglikelihood_threshold)
            break

        old_logliklihood = logliklihood

        r += 1
    return theta


def EM(n, ns_hist, transform, max_iteration, loglikelihood_threshold):
    theta = np.ones(n) / float(n)
    theta_old = np.zeros(n)
    r = 0
    sample_size = sum(ns_hist)
    old_logliklihood = 0

    while LA.norm(theta_old - theta, ord=1) > 1 / sample_size and r < max_iteration:
        theta_old = np.copy(theta)
        X_condition = np.matmul(transform, theta_old)

        TMP = transform.T / X_condition

        P = np.copy(np.matmul(TMP, ns_hist))
        P = P * theta_old

        theta = np.copy(P / sum(P))

        logliklihood = np.inner(ns_hist, np.log(np.matmul(transform, theta)))
        imporve = logliklihood - old_logliklihood

        if r > 1 and abs(imporve) < loglikelihood_threshold:
            # print("stop when", imporve, loglikelihood_threshold)
            break

        old_logliklihood = logliklihood

        r += 1
    return theta