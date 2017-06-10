import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import sys
import scipy.optimize as sco


symbols = ['intc', 'jpy', 'ms', 'erh']
noa = len(symbols)

data = pd.DataFrame()
intc = pd.read_csv('intc.csv')
intc = intc.ix[(intc['Date'] >= '2010-01-01') & (intc['Date'] <= '2016-12-31'), ['Date', 'Adj_Close']]
intc.columns = ['intc_date', 'intc_close']

jpy = pd.read_csv('jpy.csv')
jpy = jpy.ix[(jpy['DATE'] >= '2010-01-01') & (jpy['DATE'] <= '2016-12-31'), ['DATE', 'RATE']]
jpy.columns = ['jpy_date', 'jpy_rate']

total = pd.merge(intc, jpy, how='inner', left_on='intc_date', right_on='jpy_date')

ms = pd.read_csv('MS.csv')
ms = ms.ix[(ms['Date'] >= '2010-01-01') & (ms['Date'] <= '2016-12-31'), ['Date', 'Adj_Close']]
ms.columns = ['ms_date', 'ms_close']

total = pd.merge(total, ms, how='inner', left_on='intc_date', right_on='ms_date')

erh = pd.read_csv('ERH.csv')
erh = erh.ix[(erh['Date'] >= '2010-01-01') & (erh['Date'] <= '2016-12-31'), ['Date', 'Adj_Close']]
erh.columns = ['erh_date', 'erh_close']

total = pd.merge(total, erh, how='inner', left_on='intc_date', right_on='erh_date')


data['intc'] = total['intc_close']
data['jpy'] = total['jpy_rate']
data['ms'] = total['ms_close']
data['erh'] = total['erh_close']

data.columns = symbols

print data.head()
rets = np.log(data / data.shift(1))
print rets
print rets.mean()
print rets.cov()

# weights = np.random.random(noa)
# weights /= np.sum(weights)
# print weights
#
# print np.sum(rets.mean() * weights) * 252
# print np.dot(weights.T, np.dot(rets.cov() * 252, weights))
# print np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))

prets = []
pvols = []
pweights = []
for p in range(2500):
    weights = np.random.random(noa)
    weights /= np.sum(weights)
    pweights.append(weights)
    prets.append(np.sum(rets.mean() * weights) * 252)
    pvols.append(np.sqrt(np.dot(weights.T,
                            np.dot(rets.cov() * 252, weights))))
prets = np.array(prets)
pvols = np.array(pvols)

print pweights
print prets
print pvols
#
# plt.figure(figsize=(16, 8))
# plt.scatter(pvols, prets, c=prets / pvols, marker='o')
# plt.grid(True)
# plt.xlabel('expected volatility')
# plt.ylabel('expected return')
# plt.colorbar(label='Sharpe ratio')
# plt.show()

def statistics(weights):
    ''' Returns portfolio statistics.
                     Parameters
                     ==========
                     weights : array-like
                         weights for different securities in portfolio
                     Returns
                     =======
                     pret : float
                         expected portfolio return
                     pvol : float
                         expected portfolio volatility
                     pret / pvol : float
                         Sharpe ratio for rf=0
                     '''
    weights = np.array(weights)
    pret = np.sum(rets.mean() * weights) * 252
    pvol = np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))
    return np.array([pret, pvol, pret / pvol])

cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bnds = tuple((0, 1) for x in range(noa))


def min_func_variance(weights):
    return statistics(weights)[1] ** 2

optv = sco.minimize(min_func_variance, noa * [1. / noa,],
                                    method='SLSQP', bounds=bnds, constraints=cons)

print
print optv
print
print 'best optimised weight'
print optv['x'].round(3)
print 'optimised returns: ', statistics(optv['x'])[0]
print 'optimised variance: ', statistics(optv['x'])[1]

plt.figure(figsize=(16, 8))
plt.scatter(pvols, prets,
         c=prets / pvols, marker='o')
         # random portfolio composition
plt.plot(statistics(optv['x'])[1], statistics(optv['x'])[0],
      'y*', markersize=15.0)
         # minimum variance portfolio


plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
plt.show()




