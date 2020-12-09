"""
Created by Jan Schiffeler on 03.12.20
jan.schiffeler[at]gmail.com

Changed by

Selection of functions for lookback options.
Also utility functions

Python 3.7
Library version:
numpy 1.19.4
matplotlib 3.3.3

"""

import numpy as np


def generate_s(N: int, S_0: float, e_u: float, e_d: float) -> np.ndarray:
    """
    Generate a matrix with all up and down pathes (unused)
    :param N: time steps
    :param S_0: start value
    :param e_u: exp of up
    :param e_d: exp of down
    :return: matrix containing the paths
    """
    N += 1
    S_layer = np.zeros([N, N])
    S_layer[0, 0] = S_0
    for i in range(1, N):
        S_layer[:, i] = np.roll(S_layer[:, i - 1], 1) * e_d
        S_layer[0, i] = S_layer[0, i - 1] * e_u

    return S_layer


def generate_european_put_price(params: np.ndarray, N: int, K: float) -> np.ndarray:
    """

    :param params:
    :param N:
    :param K:
    :return:
    """
    pi = np.zeros(params.shape[1])
    for i in range(params.shape[1]):
        # retrieve parameters from matrix
        p, alpha, sigma, r, T, S_0 = params[:, i]
        e_d, e_u, h, q_d, q_u = unpack_parameters(N, T, alpha, p, r, sigma)

        assert q_u > 0 and q_d > 0, f"Market not arbitrage free! {q_u}; {q_d}"

        # get the payoff
        stock_price = generate_s(N=N, S_0=S_0, e_u=e_u, e_d=e_d)
        option_price = np.zeros_like(stock_price)
        option_price[:, -1] = np.maximum(K - stock_price[:, -1], 0)

        q_u = (np.exp(r * h) - e_d) / (e_u - e_d)
        q_d = (e_u - np.exp(r * h)) / (e_u - e_d)
        e_r = np.exp(r)

        for j in range(N - 1, -1, -1):
            option_price[:, j] = 1 / e_r * (q_u * option_price[:, j + 1] +
                                            q_d * np.pad(option_price[:, j + 1], (0, 1), 'constant')[1:])
        option_price = np.triu(option_price)

        # get the price at t=0 for each M samples
        pi[i] = option_price[0, 0]

    return pi


def generate_paths(M: int, N: int, S_0: float, e_u: float, e_d: float) -> [np.ndarray]:
    """
    Sample of M possible processes
    :param M: number of samples
    :param N: number of time steps
    :param S_0: start value
    :param e_u: exp of up
    :param e_d: exp of down
    :return: 1. matrix containing the possible paths of the stock price
             2. matrix containing the ups (1) and downs (0) [this is used to count N_u and N_d]
    """
    # generate a matrix of 1 and 0, where 1 = up and 0 = down of the stockprice
    instructions_pre = np.random.randint(0, 2, (M, N))
    # create a matrix with e_u and e_d entries based on the previous
    instructions = instructions_pre * e_u + (1 - instructions_pre) * e_d

    # create a matrix containing the stock price process for all M paths
    stock_price = np.zeros((M, N))
    stock_price[:, 0] = S_0
    for i in range(1, N):
        column = instructions[:, i]
        stock_price[:, i] = stock_price[:, i-1] * column

    return stock_price, instructions_pre


def calculate_payoff(stock: np.ndarray) -> np.ndarray:
    """
    Calculate the payoff of a set of paths
    :param stock:
    :return:
    """
    maximum = np.max(stock, axis=1)
    strike_price = stock[:, -1]
    return maximum - strike_price


def calculate_initial_price(params: np.ndarray, M: int, n: int, N: int=100) -> np.ndarray:
    """
    Calculate the initial for a given parameter set and for n repetitions
    :param params:  containing a set of different parameter values ->
                   |    p
                   |    alpha
                   V    sigma
                        r
                        T
                        S_0
    :param M: number of samples
    :param N: number of time steps
    :param n: number of repetitions of each experiment [test of parameter]
    :return: averaged initial price vector
    """
    pi_0 = np.zeros(params.shape[1])
    for j in range(n):
        pi_0 += get_initial_price(params=params, M=M, n=n, N=N, j=j)

    return pi_0/n


def get_initial_price(params: np.ndarray, M: int, n: int, N: int, j: int) -> np.ndarray:
    """
    Calculate the initial for a given parameter set.
    :param params:  containing a set of different parameter values ->
                   |    p
                   |    alpha
                   V    sigma
                        r
                        T
                        S_0
    :param M: number of samples
    :param N: number of time steps
    :param n: number of repetitions of each experiment [test of parameter]
    :param j: call number [only for printout]
    :return: initial price vector
    """
    print(f"Sampling Repetition {j + 1}/{n}")
    pi = np.zeros(params.shape[1])
    for i in range(params.shape[1]):
        # retrieve parameters from matrix
        p, alpha, sigma, r, T, S_0 = params[:, i]
        e_d, e_u, h, q_d, q_u = unpack_parameters(N, T, alpha, p, r, sigma)

        assert q_u > 0 and q_d > 0, f"Market not arbitrage free! {q_u}; {q_d}"

        # get the payoff for M samples of paths
        stock_paths, instructions = generate_paths(M=M, S_0=S_0, N=N, e_u=e_u, e_d=e_d)
        payoff_paths = calculate_payoff(stock=stock_paths)

        # get the price at t=0 for each M samples
        N_u = np.sum(instructions, axis=1)
        N_d = N - N_u
        pi[i] = (2 ** N) / M * np.exp(-r * h * N) * np.sum(q_u ** N_u * q_d ** N_d * payoff_paths)

    return pi


def unpack_parameters(N, T, alpha, p, r, sigma):
    h = T / N
    e_u = np.exp(alpha * h + sigma * np.sqrt(h) * np.sqrt((1 - p) / p))
    e_d = np.exp(alpha * h - sigma * np.sqrt(h) * np.sqrt(p / (1 - p)))
    q_u = (np.exp(r * h) - e_d) / (e_u - e_d)
    q_d = (e_u - np.exp(r * h)) / (e_u - e_d)
    return e_d, e_u, h, q_d, q_u


def generate_text(axis, message):
    axis.text(0, 0.5, message)
    axis.axes.xaxis.set_visible(False)
    axis.axes.yaxis.set_visible(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["bottom"].set_visible(False)
    axis.spines["left"].set_visible(False)
    return axis