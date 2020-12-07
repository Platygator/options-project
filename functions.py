"""
Created by Jan Schiffeler on 03.12.20
jan.schiffeler[at]gmail.com

Changed by



Python 3.
Library version:

TODO Set params as the third dimension of the path.
 This will affect generate_paths, calculate_payoff and get_initial_price will be freed of the loop
 Also check what is the time bottleneck of this operation
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
    instructions_pre = np.random.randint(0, 2, (M, N))
    instructions = instructions_pre * e_u + (1 - instructions_pre) * e_d
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
        p, alpha, sigma, r, T, S_0 = params[:, i]
        h = T / N
        e_u = np.exp(alpha * h + sigma * np.sqrt(h) * np.sqrt((1 - p) / p))
        e_d = np.exp(alpha * h - sigma * np.sqrt(h) * np.sqrt(p / (1 - p)))
        q_u = (np.exp(r*h) - e_d) / (e_u - e_d)
        q_d = (e_u - np.exp(r*h)) / (e_u - e_d)
        assert q_u > 0 and q_d > 0, f"Market not arbitrage free! {q_u}; {q_d}"
        stock_paths, instructions = generate_paths(M=M, S_0=S_0, N=N, e_u=e_u, e_d=e_d)
        payoff_paths = calculate_payoff(stock=stock_paths)
        N_u = np.sum(instructions, axis=1)
        N_d = N - N_u
        pi[i] = (2 ** N) / M * np.exp(-r * h * N) * np.sum(q_u ** N_u * q_d ** N_d * payoff_paths)
        # print(p, ": ", e_u, " | ", e_d)
    return pi