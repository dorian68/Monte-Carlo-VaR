# Monte-Carlo-VaR
Python script which computes the Value at risk using the Monte Carlo method

# Description
The main script is located in VaR.py
In this script we declare on instance of the class"Portfolio" and compute its VaR using the method "var_monte_carlo"

An instance of the class porfolio is made of several stocks ( class stock ) and possesses attributes suchs as variance, returns, etc, whichs represents the parameters of the portfolio. 

The returns of the differents stocks are assumed to be lognormal and by default the script runs 1000 simulations over the returns of each stock. 
We compute the VaR 95% over a one month horizon and the error of the method.
