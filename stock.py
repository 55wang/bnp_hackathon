import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import sys

symbols = ['intc', 'jpy']
noa = len(symbols)

data = pd.DataFrame()
intc = pd.read_csv('intc.csv')
intc = intc.ix[(intc['Date'] >= '2010-01-01') & (intc['Date'] <= '2016-12-31'), ['Date', 'Adj_Close']]

jpy = pd.read_csv('jpy.csv')
jpy = jpy.ix[(jpy['DATE'] >= '2010-01-01') & (jpy['DATE'] <= '2016-12-31'), ['DATE', 'RATE']]

total = pd.merge(intc, jpy, how='inner', left_on='Date', right_on='DATE')

data['intc'] = total['Adj_Close']
data['jpy'] = total['RATE']

data.columns = symbols
print data['jpy'].head()

print data.head()
rets = np.log(data / data.shift(1))
print rets
print rets.mean()
print rets.cov()

weights = np.random.random(noa)
weights /= np.sum(weights)
print weights

print np.sum(rets.mean() * weights) * 252
print np.dot(weights.T, np.dot(rets.cov() * 252, weights))
print np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))

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

plt.figure(figsize=(8, 4))
plt.scatter(pvols, prets, c=prets / pvols, marker='o')
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
plt.show()

print pweights[0][1]