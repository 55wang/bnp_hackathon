
import numpy as np
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers
import pandas as pd
import sys
np.random.seed(123)

## NUMBER OF ASSETS
n_assets = 2

## NUMBER OF OBSERVATIONS
n_obs = 10

intc = pd.read_csv('intc.csv')
jpy = pd.read_csv('jpy.csv')

intc['returns'] = intc['Adj_Close'].pct_change(1)
jpy['returns'] = jpy['RATE'].pct_change(1)

print jpy.ix[:, 'returns']
sys.exit()
return_vec = np.array([intc['returns'].values, jpy['returns'].values])
print return_vec

def rand_weights(n):
    ''' Produces n random weights that sum to 1 '''
    k = np.random.rand(n)
    return k / sum(k)

print rand_weights(n_assets)
print rand_weights(n_assets)

def random_portfolio(returns):
    ''' 
    Returns the mean and standard deviation of returns for a random portfolio
    '''

    p = np.asmatrix(np.mean(returns, axis=1))
    w = np.asmatrix(rand_weights(returns.shape[0]))
    C = np.asmatrix(np.cov(returns))

    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)

    # This recursion reduces outliers to keep plots pretty
    print 'success'
    if sigma > 2:
        return random_portfolio(returns)
    return mu, sigma

n_portfolios = 500

means, stds = np.column_stack([
    random_portfolio(return_vec)
    for _ in xrange(n_portfolios)
])

print means
print stds
