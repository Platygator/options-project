"""
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


"""


import numpy as np
import matplotlib.pyplot as plt
from functions import calculate_initial_price, generate_paths, calculate_payoff
from multiprocessing_functions import calculate_initial_price_multi


multiprocessing = False

if multiprocessing:
    calculate_price = calculate_initial_price_multi
else:
    calculate_price = calculate_initial_price

# ### Chose the task here!
# TODO might be nicer to have this in separate files

task = 2
analise = "S_0"  # p, alpha, sigma, r, T, S_0

# Base parameter set
n = 50
T = 1
p = 0.5
alpha_s = 0.1
sigma_s = 0.5
S_0 = 10
r = 0.01
N = 100

# Fixed for Task 1:
m_values = np.logspace(1, 5, num=10)

# Fixed for Task 2:
M = int(1E5)
n_2 = 10


# create ranges for the sensitivity analysis
p_range = np.arange(0.1, 0.9, 0.05)
alpha_range = np.arange(-2, 2, 0.1)
sigma_range = np.arange(0.0, 1.0, 0.1)
r_range = np.arange(0.0, 0.2, 0.05)
T_range = np.arange(1 / 12, 2, 1 / 12)
S_0_range = np.arange(5, 25, 5)

ranges = {"p": p_range, "alpha": alpha_range, "sigma": sigma_range, "r": r_range, "T": T_range, "S_0": S_0_range}
line_number = {"p": 0, "alpha": 1, "sigma": 2, "r": 3, "T": 4, "S_0": 5}
base_params = np.array([[p, alpha_s, sigma_s, r, T, S_0]]).T


# #######
# Task 1
# #######


if task == 1:
    h = T/N
    e_u = np.exp(alpha_s*h + sigma_s*np.sqrt(h)*np.sqrt((1-p)/p))
    e_d = np.exp(alpha_s*h - sigma_s*np.sqrt(h)*np.sqrt(p/(1-p)))

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

    chosen_range = ranges[analise]
    param_set = np.repeat(base_params, chosen_range.shape[0], axis=1)
    param_set[line_number[analise], :] = chosen_range

    pi = calculate_price(params=param_set, M=M, n=n_2)

    fig, ax1 = plt.subplots(1, 1)
    fig.suptitle('Task 2')
    ax1.set(title=f'Sensitivity on {analise}   |   M: {M}   n: {n_2}',
            xlabel=analise, ylabel=r'$\pi_Y(0)$',
            yscale='linear')
    ax1.label_outer()
    ax1.plot(chosen_range, pi)
    plt.show()

    print("Done")

