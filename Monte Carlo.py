# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 00:56:48 2020

@author: ld
"""

import numpy as np
from matplotlib import pyplot as plt

def monte_carlo(x,y,n):
    compt = 0
    for i in range(n):
        if(np.sqrt(x[i]**2 + y[i]**2) <= 1):
            compt += 1
    return compt/n*4

def graph():
    fig, ax = plt.subplots(1)
    ax.plot(x1, x2)
    ax.set_aspect(1)
    plt.xlim(-1.25,1.25)
    plt.ylim(-1.25,1.25)
    plt.grid(linestyle='--')
    plt.title('Monte Carlo', fontsize=8)
    plt.savefig("plot_circle_matplotlib_01.png", bbox_inches='tight')
    plt.plot(x, y)
    plt.scatter(bleux, bleuy, c = 'blue')
    plt.scatter(rougex, rougey, c = 'red')
    plt.show()



n = 10**3

theta = np.linspace(0, 2*np.pi, 100)
r = np.sqrt(1.0)

x1 = r*np.cos(theta)
x2 = r*np.sin(theta)

x = np.array([1, -1, -1, 1, 1])
y = np.array([1, 1, -1, -1, 1])

    
xclound = [np.random.uniform(-1,1) for i in range(n)]    
yclound = [np.random.uniform(-1,1) for i in range(n)]

resultat = "approx de Pi : {0} pour {1} simulations".format(monte_carlo(xclound,yclound,n),n)
print(resultat)


bleux = []
bleuy = []

rougex = []
rougey = []

for i in range(n):
    if(np.sqrt(xclound[i]**2 + yclound[i]**2)<=1):
        bleux.append(xclound[i])
        bleuy.append(yclound[i])
    else:
        rougex.append(xclound[i])
        rougey.append(yclound[i])        
    
