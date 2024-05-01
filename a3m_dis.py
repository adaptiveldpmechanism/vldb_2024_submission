# COMPUTATION OF AAA MECHANISM

import numpy as np
import scipy
from scipy import optimize

"""
xmax    # data is from [-xmax, xmax]. e.g. xmax=1
M       # total number of bins in [-xmax, xmax]. e.g. M=10
x_P     # distribution of x. e.g. np.ones(10) / 10 for uniform distribution
eps     # privacy parameter. e.g. eps=2
delta   # privacy parameter. e.g. delta=0
r       # geometric series constant. e.g. r=0.5
axRatio # the ratio amax/xmax. e.g. axRatio=2
"""

r = 0.5
def opt_variance(x_P, beta, x_bin_size, axRatio, eps, delta):
    xmax = beta
    xmin = -xmax
    M = x_P.size
    # print(M)
    x_grid = xmin + np.array(range(M)) * 0.5 * x_bin_size
    # print(x_grid)
    amax = axRatio*(np.max(x_grid)-np.min(x_grid))
    # print(amax)
    M = x_grid.size
    N = axRatio * (M-1) * 2 + 1
    # print(N)
    a_grid = np.linspace(-amax, amax, N)
    # print(a_grid)
    A = np.zeros(((M+N-1)*M*(M-1), M*N))
    counter = 0
    for k in range(-M+1, 0): # k = -M+1,...,-1
        for i in range(1, M+1): #i = 1,...,M
            for j in range(1, M+1): # j = 1,...,M
                temp = np.zeros((M, N))
                temp[i-1][max(k+i,1)-1] = np.power(r, abs(min(k+i-1,0)))
                if i != j:
                    temp[j-1][max(k+j,1)-1] = -np.exp(eps) * np.power(r, abs(min(k+j-1,0)))
                    counter = counter + 1
                    A[counter-1][:] = temp.flatten()

    for k in range(1, N-M+2): # k = 1,...,N-M+1
        for i in range(1, M+1): #i = 1,...,M
            for j in range(1, M+1): # j = 1,...,M
                temp_series = np.zeros(M)
                if i != j:
                    temp_series[i-1] = 1
                    temp_series[j-1] = -np.exp(eps)
                    diag_mat = np.zeros((M, M))
                    np.fill_diagonal(diag_mat, temp_series)
                    temp = np.concatenate((np.concatenate((np.zeros((M, k-1)), diag_mat), axis=1), np.zeros((M, N-M-k+1))), axis=1)
                    counter = counter + 1
                    A[counter-1][:] = temp.flatten()

    for k in range(N-M+1, N): # k = N-M+1,...,N-1
        for i in range(1, M+1): #i = 1,...,M
            for j in range(1, M+1): # j = 1,...,M
                temp = np.zeros((M, N))
                temp[i-1][min(k+i,N)-1] = np.power(r, abs(max(k+i-N,0)))
                if i != j:
                    temp[j-1][min(k+j,N)-1] = -np.exp(eps) * np.power(r, abs(max(k+j-N,0)))
                    counter = counter + 1
                    A[counter-1][:] = temp.flatten()

    # inequality constraints
    b = delta * np.ones((M+N-1)*(M-1)*M)

    # equality constraints
    a_matrix = np.zeros((M, M*N))
    for i in range(1, M+1):
        a_matrix[i-1][(i-1)*N : (i-1)*N+N] = a_grid

    ones_matrix = np.zeros((M, M*N))
    for i in range(1, M+1):
        ones_matrix[i-1][(i-1)*N : (i-1)*N+N] = np.ones(N)

    Aeq = np.concatenate((a_matrix, ones_matrix), axis=0)
    beq = np.concatenate((np.zeros(M), np.ones(M)), axis=0)

    Asq = np.power(a_matrix, 2)

    objMatrix = np.matmul(x_P,Asq)

    c = objMatrix.tolist()
    A_ub = A.tolist()
    b_ub = b.tolist()
    A_eq = Aeq.tolist()
    b_eq = beq.tolist()
    # print(Aeq.shape)
    # print(len(Aeq.tolist()[0]))
    # print(len(objMatrix.tolist()))

    sol = scipy.optimize.linprog(c=c,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq,b_eq=b_eq)
    # print(sol)
    # print(np.array(sol.x).size)
    # print(M)
    # print(N)
    noise_distribution = np.array(sol.x).reshape(M,N)

    return a_grid, noise_distribution

def a3m_perturb(true_histogram, beta, x_bin_size, noise_values, noise_distribution):
    M = int(2 * beta / x_bin_size) + 1
    responses = []
    for i in range(M):
        weights = np.clip(noise_distribution[i],a_min=0,a_max=None)
        if i in true_histogram:
            responses.append(np.random.choice(noise_values, true_histogram[i], True, weights/np.sum(weights)))
    responses = np.concatenate(responses, axis=None)
    return responses
