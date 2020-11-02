# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 09:41:12 2020

@author: ld
"""

from Brownian import Brownian, plot_brownian
from matplotlib import pyplot as plt
import numpy as np
import datetime as dt

S0 = 100 #initial stock price
K = 100 #strike price
r = 0.05 #risk-free interest rate
sigma = 0.20 #volatility in market
T = 1 #time in years
period = 20 #horizon of the VaR
N = 10000 #number of steps within each simulation
deltat = T/N #time step
i = 1000 #number of simulations
discount_factor = np.exp(-r*T) #discount factor


# declare a simple stock object given a ticker as name, a price at t0 (s0) which will
# be used to run simulation and a volatility named sigma
class Stock:
    def __init__(self,stock):
        self.ticker = stock['ticker']
        self.s0 = stock['price']
        self.sigma = stock['sigma']
        self.amount = stock['amount']
     
# run a simulation on the value of the stock over 20 days (variable period) using a lognormal distribution
    def run30D(self):
        returns = np.zeros(N)
        
        # define a value at t0 to base the simulation on
        price0 = Brownian(self.s0,period,r,sigma,T)[1]
        
        # run N simulations
        horizon = np.array([Brownian(price,N,r,sigma,T)[1] for price in price0])
        
        # compute the mean over the N simulation on each day of the period (20 days)
        monteCarlo = np.array([np.mean(horizon[i]) for i in range(period)])
        
        # compute the error for each day of the period ( variance )
        error = []
        tmp = []
        for i in range(len(monteCarlo)):
            tmp = np.mean([(horizon[i][j] - monteCarlo[i])**2 for j in range(len(horizon[i]))])
            error.append(tmp)
        
        error = np.array(error)        
        #t,S = Brownian(self.s0,N,r,self.sigma,T)
        S = monteCarlo
        for i in range(len(S)):
            if i == 0:
                returns[0] = 1
            else:
                returns[i] = S[i]/S[i - 1]
        return S, returns, error
    
    def plot(self):
        plot_brownian(self.s0,N,self.sigma,T)

class Portfolio:
    Sum = 0
    stocklst = []
    returns = []
    variance = 0
    amount = 0
    value = 0 
    def __init__(self,*stocks):
        for stk in stocks:
            self.stocklst.append(stk)
            self.amount += stk.amount
            self.value += stk.s0 * stk.amount
    
    def ptf_sum(self):
        return self.Sum
        
    def sorted_returns(self):
        ptfSum = np.zeros(period)
        sumOfReturn = np.zeros(period)
            
        for stk in self.stocklst:
            S, returns, error = stk.run30D()
            self.Sum += S * stk.amount
            self.variance += error
            ptfSum += S
        for i in range(len(ptfSum)):
            if i == 0:
                sumOfReturn[0] = 1
            else:
                sumOfReturn[i] = ptfSum[i]/ptfSum[i - 1]
        sumOfReturn = np.sort(sumOfReturn - 1 ,kind='mergesort')
        self.returns = np.array([i for i in sumOfReturn])
        return self.returns
    
    def var_monte_carlo(self):
        return np.percentile(self.returns,5,axis=0) * self.value
    
    def error_monte_carlo(self):
        return self.variance


begin = dt.datetime.now()

# define the dictionnary which will be used to stock instances
stock = {
    "ticker": "apple",
    "price": 120,
    "sigma": 0.20,
    "amount": 30,
    }

stock2 = {
    "ticker": "microsoft",
    "price": 98,
    "sigma": 0.09,
    "amount": 50,
    }

stock3 = {
    "ticker": "goldman sachs",
    "price": 220,
    "sigma": 0.1,
    "amount": 70,
    }

stock4 = {
    "ticker": "compagnie X",
    "price": 370,
    "sigma": 0.07,
    "amount": 70,
    }

stock5 = {
    "ticker": "compagnie y",
    "price": 175,
    "sigma": 0.16,
    "amount": 220,
    }

# declare stock instances
apple = Stock(stock)
microsoft = Stock(stock2)
goldman = Stock(stock3)
compagniex = Stock(stock4)
compagniey = Stock(stock5)

ptf = Portfolio(apple,microsoft,goldman,compagniex,compagniey)
data = ptf.sorted_returns()
var = ptf.var_monte_carlo()
error = ptf.error_monte_carlo()
value = ptf.ptf_sum()

end = dt.datetime.now()

print("-----------------------------------------------------------------------")
print("value at t0: ",value)
print("-----------------------------------------------------------------------")
print("var monte carlo: ",var)
print("-----------------------------------------------------------------------")
print("error: ",np.mean(error))
print("-----------------------------------------------------------------------")
print(end - begin)
print("-----------------------------------------------------------------------")

plt.plot(list(range(period)),value)


price0 = Brownian(S0,20,r,sigma,T)[1]
horizon = np.array([Brownian(price,N,r,sigma,T)[1] for price in price0])