"""
Created by Jan Schiffeler on 03.12.20
jan.schiffeler[at]gmail.com

Changed by



Python 3.
Library version:


"""
import numpy as np
from multiprocessing import Process, Manager, cpu_count

from functions import get_initial_price


def calculate_initial_price_multi(params, M, n, N=100):
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
    :return:
    """
    manager = Manager()
    experiment_dict = manager.dict()

    procs = []

    assert n < cpu_count(), f"You need more cores than n! #Cores found: {cpu_count()} | n: {n}"
    for j in range(n):
        proc = Process(target=worker_func,
                       kwargs={"M": M, "N": N, "params": params, "j": j, "n": n, "experiment_dict": experiment_dict})
        procs.append(proc)
        proc.start()

    pi_0 = np.zeros(params.shape[1])

    for proc in procs:
        proc.join()

    for key, value in experiment_dict.items():
        # print(key, ": ", value)
        pi_0 += value

    return pi_0/n

# def process_pi_0_multi(func, experiment_dict: dict, params: np.ndarray, M: int, n: int, N: int, j: int) -> np.ndarray:
#     def wrapped_multi(experiment_dict: dict, params: np.ndarray, M: int, n: int, N: int, j: int):
#         pi_0 = func(params=params, M=M, n=n, N=N, j=j)
#         experiment_dict[j] = pi_0
#
#     return wrapped_multi


def worker_func(experiment_dict: dict, params: np.ndarray, M: int, n: int, N: int, j: int):
    """

    :param experiment_dict: joined data collector for all parallel processes
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
    :param j: call number
    :return: write to shared dictionary
    """
    np.random.seed()
    pi_0 = get_initial_price(params=params, M=M, n=n, N=N, j=j)
    experiment_dict[j] = pi_0
