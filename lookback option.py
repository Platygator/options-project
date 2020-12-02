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
analise = "alpha"  # p, alpha, sigma, r, T, S

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


def get_y_0(params, M, N=100):
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

    # TODO also use a small n
    pi_0 = []
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
        pi_0.append(pi)
    return pi_0

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
    p_n = p
    alpha_n = alpha_s
    sigma_n = sigma_s
    r_n = r
    T_n = T
    S_0_n = S_0

    p_range = np.arange(0.1, 0.9, 0.1)
    alpha_range = np.arange(-2, 2, 0.1)
    sigma_range = np.arange(0.0, 1.0, 0.1)
    r_range = np.arange(0.0, 0.2, 0.05)
    T_range = np.arange(1/12, 2, 1/12)
    S_0_range = np.arange(5, 25, 5)

    pi = 0
    chosen_range = 0
    
    # p sensitivity analysis
    if analise == "p":
        p_set = np.ones((6, p_range.shape[0]))
        p_set[0, :] = p_range
        p_set[1, :] *= alpha_n
        p_set[2, :] *= sigma_n
        p_set[3, :] *= r_n
        p_set[4, :] *= T_n
        p_set[5, :] *= S_0_n
        pi_p = get_y_0(p_set, 100)

        pi = pi_p
        chosen_range = p_range
    
    # p sensitivity analysis
    elif analise == "alpha":
        alpha_set = np.ones((6, alpha_range.shape[0]))
        alpha_set[0, :] *= p_n
        alpha_set[1, :] = alpha_range
        alpha_set[2, :] *= sigma_n
        alpha_set[3, :] *= r_n
        alpha_set[4, :] *= T_n
        alpha_set[5, :] *= S_0_n
        pi_alpha = get_y_0(alpha_set, 100)

        pi = pi_alpha
        chosen_range = alpha_range

    # sigma sensitivity analysis
    elif analise == "sigma":
        sigma_set = np.ones((6, sigma_range.shape[0]))
        sigma_set[0, :] *= p_n
        sigma_set[1, :] *= alpha_n
        sigma_set[2, :] = sigma_range
        sigma_set[3, :] *= r_n
        sigma_set[4, :] *= T_n
        sigma_set[5, :] *= S_0_n
        pi_sigma = get_y_0(sigma_set, 100)

        pi = pi_sigma
        chosen_range = sigma_range    
        
    # r sensitivity analysis
    elif analise == "r":
        r_set = np.ones((6, r_range.shape[0]))
        r_set[0, :] *= p_n
        r_set[1, :] *= alpha_n
        r_set[2, :] *= sigma_n
        r_set[3, :] = r_range
        r_set[4, :] *= T_n
        r_set[5, :] *= S_0_n
        pi_r = get_y_0(r_set, 100)

        pi = pi_r
        chosen_range = r_range

    # T sensitivity analysis
    elif analise == "T":
        T_set = np.ones((6, T_range.shape[0]))
        T_set[0, :] *= p_n
        T_set[1, :] *= alpha_n
        T_set[2, :] *= sigma_n
        T_set[3, :] *= r_n
        T_set[4, :] = T_range
        T_set[5, :] *= S_0_n
        pi_T = get_y_0(T_set, 100)

        pi = pi_T
        chosen_range = T_range

    # S_0 sensitivity analysis
    elif analise == "S_0":
        S_0_set = np.ones((6, S_0_range.shape[0]))
        S_0_set[0, :] *= p_n
        S_0_set[1, :] *= alpha_n
        S_0_set[2, :] *= sigma_n
        S_0_set[3, :] *= r_n
        S_0_set[4, :] *= T_n
        S_0_set[5, :] = S_0_range
        pi_S_0 = get_y_0(S_0_set, 100)

        pi = pi_S_0
        chosen_range = S_0_range

    fig, ax1 = plt.subplots(1, 1)
    fig.suptitle('Task 2')
    ax1.set(title=f'Sensitivity on {analise}',
            xlabel=analise, ylabel=r'$\pi_Y(0)$',
            yscale='linear')
    ax1.label_outer()
    ax1.plot(chosen_range, pi)
    plt.show()

    print("Done")

