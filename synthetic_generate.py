import numpy as np
import time

def generate_synthetic_data(data_type, n, low, high, beta):
    # generate data
    if data_type == "GAUSSIAN":
        data = np.random.normal(0, 1.0, n)
        # print(np.std(data))
    elif data_type == "EXPONENTIAL":
        data = np.random.exponential(1, n)
        low = 0
    elif data_type == "GAMMA":
        data = np.random.gamma(2, 2, n)
        low = 0
    elif data_type == "BETA":
        data = np.random.beta(0.5, 0.5, n)
        low = 0
        high = 1
    elif data_type == "UNIFORM":
        data = np.random.uniform(0, 1, n)
    elif data_type == "GAUSSMIX":
        data_1 = np.random.normal(-1, 0.5, n)
        data_2 = np.random.normal(1, 0.5, n)
        p = 0.5 * np.ones(n) 
        ber = np.random.binomial(1, p, size=None)
        data = ber*data_1 + (1-ber)*data_2
    elif data_type == "BER":
        p = 0.5 * np.ones(n) 
        low = 0
        high = 1
        data = np.random.binomial(1, p, size=None)
    else:
        raise NotImplementedError()
    # clip data to [low,high]
    clipped_data = np.clip(data, low, high)
    # linearly map data from [low,high] to [-beta,beta]
    """ Let linear map be: y = ax + b
        Then: 
        -beta = a*low + b and beta = a*high+b
        We have:
    """
    a = 2*beta / (high-low)
    b = beta - 2*beta*high / (high-low)
    mapped_data = a*clipped_data + b
    return mapped_data