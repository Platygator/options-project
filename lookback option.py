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


TODO turn the functions into generators to deal with the huge M matrices
"""


import numpy as np
import matplotlib.pyplot as plt

task = 2
analise = "S_0"  # p, alpha, sigma, r, T, S_0

# Fixed for Task 1:
n = 50
T = 1
p = 0.5
alpha_s = 0.1
sigma_s = 0.5
S_0 = 10
r = 0.01
N = 100

# Fixed for Task 2:
M = int(10)
n_2 = 10

p_range = np.arange(0.1, 0.9, 0.1)
alpha_range = np.arange(-2, 2, 0.1)
sigma_range = np.arange(0.0, 1.0, 0.1)
r_range = np.arange(0.0, 0.2, 0.05)
T_range = np.arange(1 / 12, 2, 1 / 12)
S_0_range = np.arange(5, 25, 5)

ranges = {"p": p_range, "alpha": alpha_range, "sigma": sigma_range, "r": r_range, "T": T_range, "S_0": S_0_range}
line_number = {"p": 0, "alpha": 1, "sigma": 2, "r": 3, "T": 4, "S_0": 5}
base_params = np.array([[p, alpha_s, sigma_s, r, T, S_0]]).T


def generate_s(N, S_0, e_u, e_d):
    N += 1
    S_layer = np.zeros([N, N])
    S_layer[0, 0] = S_0
    for i in range(1, N):
        S_layer[:, i] = np.roll(S_layer[:, i - 1], 1) * e_d
        S_layer[0, i] = S_layer[0, i - 1] * e_u

    return S_layer


def generate_paths(S_0, M, N, e_u, e_d):
    instructions_pre = np.random.randint(0, 2, (M, N))
    instructions = instructions_pre * e_u + (1 - instructions_pre) * e_d
    stock_price = np.zeros((M, N))
    stock_price[:, 0] = S_0
    for i in range(1, N):
        slice = instructions[:, i]
        stock_price[:, i] = stock_price[:, i-1] * slice

    return stock_price, instructions_pre


def generate_payoff(stock):
    maximum = np.max(stock, axis=1)
    strike_price = stock[:, -1]
    return maximum - strike_price


def get_y_0(params, M, n, N=100):
    """

    :param params:      p
                        alpha
                        sigma
                        r
                        T
                        S_0
    :param M:
    :param N:
    :return:
    """

    pi_0 = np.zeros(params.shape[1])
    for j in range(n):
        for i in range(params.shape[1]):
            p, alpha, sigma, r, T, S_0 = params[:, i]
            h = T / N
            e_u = np.exp(alpha_s * h + sigma_s * np.sqrt(h) * np.sqrt((1 - p) / p))
            e_d = np.exp(alpha_s * h - sigma_s * np.sqrt(h) * np.sqrt(p / (1 - p)))
            q_u = (np.exp(r) - e_d) / (e_u - e_d)
            q_d = 1 - q_u
            # TODO do we have to make an arbitrage free check here?
            stock_paths, instructions = generate_paths(S_0, M, N, e_u, e_d)
            payoff_paths = generate_payoff(stock_paths)
            N_u = np.sum(instructions, axis=1)
            N_d = N - N_u
            pi = (2**N)/M * np.exp(-r * h * N) * np.sum(q_u**N_u * q_d**N_d * payoff_paths)
            pi_0[i] += pi

    return pi_0/n

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

    m_values = [k for k in range(100, 100000, 10000)]
    for m in m_values:
        mean = 0
        error = 0
        for i in range(n):
            stock_paths, _ = generate_paths(S_0, m, N, e_u, e_d)
            payoff_paths = generate_payoff(stock_paths)

            current_mean = np.mean(payoff_paths)
            mean += current_mean
            current_error = np.sqrt( 1/(n-1) * np.sum((payoff_paths - current_mean)**2) )/np.sqrt(n)
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
    ax1.plot(mean_store)
    ax2.plot(error_store)
    plt.show()


# #######
# Task 2
# #######

elif task == 2:

    chosen_range = ranges[analise]
    param_set = np.repeat(base_params, chosen_range.shape[0], axis=1)
    param_set[line_number[analise], :] = chosen_range

    pi = get_y_0(param_set, 100, n_2)

    fig, ax1 = plt.subplots(1, 1)
    fig.suptitle('Task 2')
    ax1.set(title=f'Sensitivity on {analise}',
            xlabel=analise, ylabel=r'$\pi_Y(0)$',
            yscale='linear')
    ax1.label_outer()
    ax1.plot(chosen_range, pi)
    plt.show()

    print("Done")

