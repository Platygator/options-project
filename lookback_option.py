"""
Created by Jan Schiffeler and Chengji Wang on 01.12.20

Options and mathematics Project
Lookback Put Derivative


Task 1:
Influence on Err and Mean
    of M

Task 2:
Influences on Pi_y(0)
    of p, alpha, sigma, r, T, S_0

Task 3:
Compare to European Put


Python 3.7
Library version:
numpy 1.19.4
matplotlib 3.3.3
"""


import numpy as np
import matplotlib.pyplot as plt
from functions import calculate_initial_price, generate_paths, calculate_payoff, generate_european_put_price
from multiprocessing_functions import calculate_initial_price_multi

# usually multiprocessing is faster, but might create problems
# if the available number of cores is lower than I expect
multiprocessing = True

if multiprocessing:
    calculate_price = calculate_initial_price_multi
else:
    calculate_price = calculate_initial_price

# ### #################### ###
# ### Chose the task here! ###
# ### #################### ###

task = 3
# for task 2 also set a parameter to be tested
analise = "S_0"  # p, alpha, sigma, r, T, S_0

# ### #################### ###
# ### Chose the task here! ###
# ### #################### ###

# Base parameter set
n = 50
N = 100

p = 0.5
alpha = 0.1
sigma = 0.5
r = 0.01
T = 1
S_0 = 10

# Fixed for Task 1:
m_values = np.logspace(1, 5, num=10)

# Fixed for Task 2:
M = int(1E5)
n_2 = 20

# Fixed for Task 3:
K = 10
M = 100


# create ranges for the sensitivity analysis for task 2
p_range = np.arange(0.1, 0.9, 0.05)
alpha_range = np.arange(-2, 2, 0.1)
sigma_range = np.arange(0.1, 1.0, 0.1)
r_range = np.arange(0.0, 0.2, 0.05)
T_range = np.arange(1/12, 2, 1/12)
S_0_range = np.arange(5, 25, 2)

# create dictionaries for easy switching between parameter analyses
ranges = {"p": p_range, "alpha": alpha_range, "sigma": sigma_range, "r": r_range, "T": T_range, "S_0": S_0_range}
line_number = {"p": 0, "alpha": 1, "sigma": 2, "r": 3, "T": 4, "S_0": 5}
base_params = np.array([[p, alpha, sigma, r, T, S_0]]).T


# #######
# Task 1
# #######

if task == 1:
    h = T/N
    e_u = np.exp(alpha*h + sigma*np.sqrt(h)*np.sqrt((1-p)/p))
    e_d = np.exp(alpha*h - sigma*np.sqrt(h)*np.sqrt(p/(1-p)))

    # fill out S
    timesteps = np.arange(0, 1 + T/N, T/N)

    mean_store = []
    error_store = []

    for m in m_values:
        print(f"Processing M: {m}")
        m = int(m)
        mean = 0
        error = 0
        for i in range(n):
            stock_paths, _ = generate_paths(M=m, S_0=S_0, N=N, e_u=e_u, e_d=e_d)
            payoff_paths = calculate_payoff(stock=stock_paths)

            current_mean = np.mean(payoff_paths)
            mean += current_mean
            current_error = np.sqrt( 1/(m-1) * np.sum((payoff_paths - current_mean)**2) )/np.sqrt(m)
            error += current_error

        mean_store.append(mean/n)
        error_store.append(error/n)
    
    # plotting
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('Task 1')
    ax1.set(title='Mean',
            xlabel='M', ylabel='mean',
            yscale='linear')
    ax1.label_outer()
    ax2.set(title='Standard Error',
            xlabel='M', ylabel='Standard Error',
            yscale='linear')
    ax2.label_outer()
    ax1.plot(m_values, mean_store)
    ax2.plot(m_values, error_store)
    plt.show()


# #######
# Task 2
# #######

elif task == 2:
    
    """
    Build a parameter matrix to test the different scenarios.
    
    If e.g. p is supposed to be tested the params matrix could look like this:
    
    p      | 0.1, 0.2, 0.3, 0.4, ....
    alpha  | 0.1, 0.1, 0.1, 0.1, ....
    sigma  | 0.1, 0.1, 0.1, 0.1, ....
    r      | 0.1, 0.1, 0.1, 0.1, ....
    T      | 1.0, 1.0, 1.0, 1.0, ....
    S_0    |  10,  10,  10, 1 0, ....
    
    """
    chosen_range = ranges[analise]
    param_set = np.repeat(base_params, chosen_range.shape[0], axis=1)
    param_set[line_number[analise], :] = chosen_range
    
    # run two times. This is separated as I use n as the number of processes running 
    # in parallel and I only have 12 cores
    pi = calculate_price(params=param_set, M=M, n=int(n_2/2))
    pi += calculate_price(params=param_set, M=M, n=int(n_2/2))
    pi /= 2
    
    # plotting
    fig, ax1 = plt.subplots(1, 1)
    fig.suptitle('Task 2')
    ax1.set(title=f'Sensitivity on {analise}   |   M: {M}   n: {n_2}',
            xlabel=analise, ylabel=r'$\pi_Y(0)$',
            yscale='linear')
    ax1.label_outer()
    ax1.plot(chosen_range, pi)
    plt.show()


# #######
# Task 3
# #######

elif task == 3:

    # TODO What is on x? For European only the last layer?

    # calculate European put price
    h = T/N
    e_u = np.exp(alpha*h + sigma*np.sqrt(h)*np.sqrt((1-p)/p))
    e_d = np.exp(alpha*h - sigma*np.sqrt(h)*np.sqrt(p/(1-p)))

    pi_euro = generate_european_put_price(N=N, S_0=S_0, e_u=e_u, e_d=e_d, r=r, h=h, K=K)

    # calculate Lookback option price
    pi_asia = calculate_price(params=base_params, M=M, n=int(n_2/2))
    pi_asia += calculate_price(params=base_params, M=M, n=int(n_2/2))
    pi_asia /= 2

    # plotting
    fig, ax1 = plt.subplots(1, 1)
    fig.suptitle('Task 3')
    ax1.set(title=f'Comparision between standard European put and floating strike Asian Lookback put option',
            xlabel="Yes. Dunno", ylabel=r'$\pi_Y(0)$',
            yscale='linear')
    ax1.label_outer()
    ax1.plot(pi_euro)
    ax1.plot(pi_asia)
    plt.show()

print("Finished")
