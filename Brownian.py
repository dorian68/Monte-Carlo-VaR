# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 10:16:58 2020

@author: ld
"""

import numpy as np
from matplotlib import pyplot as plt


S0 = 100 #initial stock price
K = 100 #strike price
r = 0.05 #risk-free interest rate
sigma = 0.20 #volatility in market
T = 1 #time in years
N = 100 #number of steps within each simulation
deltat = T/N #time step
i = 1000 #number of simulations
discount_factor = np.exp(-r*T) #discount factor

S = np.zeros([i,N])
t = range(0,N,1)


def Monte_carlo(i,S0,N,sigma,deltat):
    for y in range(0,i-1):
        S[y,0]=S0
        for x in range(0,N-1):
            S[y,x+1] = S[y,x]*(np.exp((r-(sigma**2)/2)*deltat + sigma*deltat*np.random.normal(0,1)))
        plt.plot(t,S[y])
    
    plt.title('Simulations %d Steps %d Sigma %.2f r %.2f S0 %.2f' % (i, N, sigma, r, S0))
    plt.xlabel('Steps')
    plt.ylabel('Stock Price')
    plt.show()
    
    C = np.zeros((i-1,1), dtype=np.float16)
    for y in range(0,i-1):
        C[y]=np.maximum(S[y,N-1]-K,0)
    
def Brownian(S0,N,r,sigma,T):
    deltat = T/N
    S = np.zeros([N])
    t = range(0,N,1)
    for y in range(0,i-1):
        S[0] = S0
    for x in range(0,N-1):
        S[x+1] = S[x]*(np.exp((r-(sigma**2)/2)*deltat + sigma*deltat*np.random.normal(0,1)))
    return t,S
    
    

def plot_brownian(S0,N,sigma,T):
    t,S = Brownian(S0,N,r,sigma,T)
    plt.title('Asset price simulation using Brownian motion')  
    plt.plot(t,S)
    plt.savefig('Share100.jpg')
    plt.show()
    
#CallPayoffAverage = np.average(C)
#CallPayoff = discount_factor*CallPayoffAverage
#print(CallPayoff)