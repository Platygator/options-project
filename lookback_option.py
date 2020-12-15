"""
Created by Chengji Wang and Jan Schiffeler on 01.12.20

Options and mathematics Project
Lookback Put Derivative


Task 1:
Influence on Err and Mean of the initial price
    of M

Task 2:
Influences on Pi_y(0)
    of p, alpha, sigma, r, T, S_0

Task 3:
Comparison to European Put [incorporated in Task 2]


Python 3.7
Library version:
numpy 1.19.4
matplotlib 3.3.3

"""

import numpy as np
from matplotlib import pyplot as plt
from functions import calculate_initial_price, get_initial_price, generate_european_put_price
from functions import generate_text
from multiprocessing_functions import calculate_initial_price_multi

# usually multiprocessing is faster up until and including M=1E5
multiprocessing = True
# if set higher than the available number of threads the program will stop
# if set very close to the max number the computer will start to lag.
# then abort with Ctrl+C or wait
cores = 5

if multiprocessing:
    calculate_price = calculate_initial_price_multi
else:
    calculate_price = calculate_initial_price

# ### #################### ###
# ### Chose the task here! ###

task = 1
# for task 2 also set a parameter to be tested
analise = "r"  # p, alpha, sigma, r, T, S_0


# ### #################### ###

# Base parameter set
n = 50
N = 100

p = 0.5
alpha = 0.0
sigma = 0.3
r = 0.01
T = 1
S_0 = 10

# Fixed for Task 1:
m_values = np.logspace(1, 5, num=10)

# Fixed for Task 2:
M = int(1E5)
n_2 = 50

# Fixed for Task 3:
K = 20


# create ranges for the sensitivity analysis for task 2
p_range = np.arange(0.025, 1.0, 0.025)
alpha_range = np.arange(-2, 2, 0.1)
sigma_range = np.arange(0.5, 10.5, 1)
r_range = np.arange(0.0, 0.51, 0.1)
T_range = np.arange(1/12, 2 + 1/12, 1/6)
S_0_range = np.arange(5, 25, 2)

# create dictionaries for easy switching between parameter analyses
ranges = {"p": p_range, "alpha": alpha_range, "sigma": sigma_range, "r": r_range, "T": T_range, "S_0": S_0_range}
line_number = {"p": 0, "alpha": 1, "sigma": 2, "r": 3, "T": 4, "S_0": 5}
latex_string = {"p": "p", "alpha": r"$\alpha$", "sigma": r"$\sigma$", "r": "r", "T": "T [years]", "S_0": r"$S_0$"}
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

    for j, m in enumerate(m_values):
        print(f"Processing M: {m}")
        m = int(m)
        # Note: this function is designed to the efficient calculation of task 2 and thus is used in a bit
        # odd fashion here.
        pi_0 = get_initial_price(params=np.repeat(base_params, n, axis=1), M=m, n=1, N=N, j=j)

        mean = np.mean(pi_0)
        error = np.sqrt(1/(m-1) * np.sum((pi_0 - mean)**2))/np.sqrt(m)

        mean_store.append(mean)
        error_store.append(error)
    
    # plotting
    fig, ((ax1, text1), (ax2, text2)) = plt.subplots(2, 2, gridspec_kw={'width_ratios': [2, 1]})
    fig.suptitle('Task 1')
    ax1.set(title='Mean',
            xlabel='M', ylabel='mean',
            yscale='linear', xscale='log')
    ax1.label_outer()
    ax2.set(title='Standard Error',
            xlabel='M', ylabel='standard error',
            yscale='linear', xscale='log')
    ax2.label_outer()
    ax1.plot(m_values, mean_store)
    ax2.plot(m_values, error_store)
    message_a = f"Parameters in this plot: \n" \
                f"M: {M}\n" \
                f"n: {n}\n" \
                f"p: {p}\n"
    message_b = r"$\alpha$: " + f"{alpha}\n" + \
                r"$\sigma$: " + f"{sigma}\n" + \
                f"r: {r}\n" \
                f"T: {T}\n" \
                f"S_0: {S_0}\n"
    text1 = generate_text(text1, message_a)
    text2 = generate_text(text2, message_b)

    plt.savefig(f"plots/task1.png", format="png")
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

    # European option price
    pi_euro = generate_european_put_price(params=param_set, N=N, K=K)

    # Lookback option price
    pi = calculate_price(params=param_set, M=M, n=cores)
    reps = int(n_2/cores)
    for i in range(1, reps):
        print("Set number: ", i+1, "/", reps)
        pi += calculate_price(params=param_set, M=M, n=cores)
    pi /= reps

    # plotting
    fig, (ax1, text) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]})
    fig.suptitle('Task 2')
    ax1.set(title=f'Sensitivity on {latex_string[analise]}',
            xlabel=latex_string[analise], ylabel=r'$\pi_Y(0)$',
            yscale='linear')
    ax1.label_outer()
    ax1.plot(chosen_range, pi, label="Lookback", color=(0.59, 0.11, 0.19))

    ax1b = ax1.twinx()
    ax1b.plot(chosen_range, pi_euro, label="European", color=(0.35, 0.35, 0.87))

    message = f"Parameters in this plot: \n" \
              f"M: {M}\n" \
              f"n: {n_2}\n" \
              f"p: {p}\n" \
              r"$\alpha$: " + f"{alpha}\n" + \
              r"$\sigma$: " + f"{sigma}\n" + \
              f"r: {r}\n" \
              f"T: {T}\n" \
              r"$S_0$: " + f"{S_0}\n" + \
              f"K: {K}"

    text = generate_text(text, message)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    text.legend(lines + lines2, labels + labels2, loc=(0, 0.2))

    fig.tight_layout()

    plt.savefig(f"plots/task2_{analise}.png", format="png")
    plt.show()


print("Finished")
