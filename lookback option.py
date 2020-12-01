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


# Fixed for Task 1:
n = 50
T = 1
p = 0.5
alpha_s = 0.1
sigma_s = 0.5
S_0 = 10
N = 100
r = 0.01

# Fixed for Task 2:
M = int(10)
N = 100


def generate_s(N, S_0, e_u, e_d):
    N += 1
    S_layer = np.zeros([N, N])
    S_layer[0, 0] = S_0
    for i in range(1, N):
        S_layer[:, i] = np.roll(S_layer[:, i - 1], 1) * e_d
        S_layer[0, i] = S_layer[0, i - 1] * e_u

    return S_layer


def generate_paths(S_0, M, N, e_u, e_d):
    instructions = np.random.choice((e_u, e_d), (M, N))
    stock_price = np.zeros((M, N))
    stock_price[:, 0] = S_0
    for i in range(1, N):
        slice = instructions[:, i]
        stock_price[:, i] = stock_price[:, i-1] * slice

    return stock_price


def generate_payoff(stock):
    maximum = np.max(stock, axis=1)
    strike_price = stock[:, -1]
    return maximum - strike_price

# #######
# Task 1
# #######

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
        stock_paths = generate_paths(S_0, m, N, e_u, e_d)
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



#Rand,Nu = RandomPathsBinomial(S,50)

#plt.plot(np.transpose(Rand)) #as the arrays are column vectors not row
#plt.show()

# P1 = np.zeros(100)
# P2 = np.zeros(100)
# P3 = np.zeros(100)
#
# for i in range(100):
#     P1[i] = BinomialLookback(Q,S,0.01,100)
#     P2[i] = BinomialLookback(Q,S,0.01,1000)
#     P3[i] = BinomialLookback(Q,S,0.01,10000)
#
# plt.figure(2)
# plt.plot(P1)
# plt.plot(P2)
# plt.plot(P3)
# plt.show()
