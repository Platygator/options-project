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
import math



# Fixed for Task 1:
n = 50
T = 1
p = 0.5
alpha = 0.1
sigma = 0.5
S_0 = 10
N = 100

# Fixed for Task 2:
N = 100
M = 10E5


def BinomialStock(p, alpha, sigma, s, T, N):
    h = T/N
    u = alpha*h + sigma*np.sqrt(h)*np.sqrt((1-p)/p)
    d = alpha*h - sigma*np.sqrt(h)*np.sqrt(p/(1-p))
    Q = np.zeros(N+1)
    S = np.zeros((N+1, N+1))
    Q[0] = 0
    S[0, 0] = s
    for j in range(N):
        Q[j+1] = (j+1) * h
        S[0, j+1] = S[0, j]*np.exp(u)
        for i in range(j+1):
            S[i+1, j+1] = S[i, j]*np.exp(d)
    return Q, S


def RandomPathsBinomial(S, M):
    N = len(S)-1
    r = np.random.randint(0,2,size=(M,N))
    Nu = np.sum(r==0,axis=1)
    Rp = np.zeros((M,N+1))
    rows = np.zeros((M,N+1))
    rows[:,0] = 0
    Rp[:,0] = S[0,0]
    for j in range(N):
        rows[:,j+1] = rows[:,j] + r[:,j]
        Rp[:,j+1]= S[np.int_(rows[:,j+1]),j+1]
    return Rp,Nu


def BinomialLookback(Q,S,r,M):
    h = Q[2]-Q[1]
    N = len(Q)-1
    expu = S[0,1]/S[0,0]
    expd = S[1,1]/S[0,0]  #goes down rate
    qu = (np.exp(r*h)-expd)/(expu-expd)
    qd = (expu-np.exp(r*h))/(expu-expd)
    if (qu<0) | (qd<0):
        print('Error: the market is not arbitrage free.')
        price = 0
        return
    [R,Nu] = RandomPathsBinomial(S,M)      #R is M*(N+1), Nu is size M vector
    payoff = np.max(R, axis = 1)- R[:,N]
    terms = (qu**Nu)*(qd**(N-Nu))*payoff
    price = np.exp(-r*h*N)*sum(terms)*2**N/M    #monte carlo:2^N the possible paths. M samples
    return price

# #######
# Task 1
# #######

Q, S = BinomialStock(p, alpha, sigma, S_0, T, N)


#Rand,Nu = RandomPathsBinomial(S,50)

#plt.plot(np.transpose(Rand)) #as the arrays are column vectors not row
#plt.show()

P1 = np.zeros(100)
P2 = np.zeros(100)
P3 = np.zeros(100)

for i in range(100):
    P1[i] = BinomialLookback(Q,S,0.01,100)
    P2[i] = BinomialLookback(Q,S,0.01,1000)
    P3[i] = BinomialLookback(Q,S,0.01,10000)

plt.figure(2)
plt.plot(P1)
plt.plot(P2)
plt.plot(P3)
plt.show()
