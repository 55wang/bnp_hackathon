import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import sys
import scipy.optimize as sco


# initial_invest_amt = 10000
# initial_invest_amt = 20000
initial_invest_amt = 50000

path = 'csv/'
symbols = ['intc', 'jpy', 'ms', 'erh', 'med', 'barl', 'STAG_INDUSTRIAL_INC', 'DDR_CORP', 'EQUITY_RESIDENTIAL_PROPERTIES']
instrument_value = [548, 159, 292391, 378790, 1139292, 1072180, 1123568, 1140653, 29665]

noa = len(symbols)

data = pd.DataFrame()
intc = pd.read_csv(path + '548.csv')
intc = intc.ix[(intc['Date'] >= '2010-01-01') & (intc['Date'] <= '2016-12-31'), ['Date', 'Adj_Close']]
intc.columns = ['intc_date', 'intc_close']

jpy = pd.read_csv(path + 'jpy.csv')
jpy = jpy.ix[(jpy['DATE'] >= '2010-01-01') & (jpy['DATE'] <= '2016-12-31'), ['DATE', 'RATE']]
jpy.columns = ['jpy_date', 'jpy_rate']

total = pd.merge(intc, jpy, how='inner', left_on='intc_date', right_on='jpy_date')

ms = pd.read_csv(path + 'MS.csv')
ms = ms.ix[(ms['Date'] >= '2010-01-01') & (ms['Date'] <= '2016-12-31'), ['Date', 'Adj_Close']]
ms.columns = ['ms_date', 'ms_close']

total = pd.merge(total, ms, how='inner', left_on='intc_date', right_on='ms_date')

erh = pd.read_csv(path + 'ERH.csv')
erh = erh.ix[(erh['Date'] >= '2010-01-01') & (erh['Date'] <= '2016-12-31'), ['Date', 'Adj_Close']]
erh.columns = ['erh_date', 'erh_close']

total = pd.merge(total, erh, how='inner', left_on='intc_date', right_on='erh_date')

med = pd.read_csv(path + 'MED.csv')
med = med.ix[(med['Date'] >= '2010-01-01') & (med['Date'] <= '2016-12-31'), ['Date', 'Adj_Close']]
med.columns = ['med_date', 'med_close']

total = pd.merge(total, med, how='inner', left_on='intc_date', right_on='med_date')

barl = pd.read_csv(path + 'barl.csv')
barl = barl.ix[(barl['Date'] >= '2010-01-01') & (barl['Date'] <= '2016-12-31'), ['Date', 'Adj_Close']]
barl.columns = ['barl_date', 'barl_close']

total = pd.merge(total, barl, how='inner', left_on='intc_date', right_on='barl_date')

STAG_INDUSTRIAL_INC = pd.read_csv(path + 'STAG_INDUSTRIAL_INC.csv')
STAG_INDUSTRIAL_INC = STAG_INDUSTRIAL_INC.ix[(STAG_INDUSTRIAL_INC['Date'] >= '2010-01-01') & (STAG_INDUSTRIAL_INC['Date'] <= '2016-12-31'), ['Date', 'Adj_Close']]
STAG_INDUSTRIAL_INC.columns = ['STAG_INDUSTRIAL_INC_date', 'STAG_INDUSTRIAL_INC_close']

total = pd.merge(total, STAG_INDUSTRIAL_INC, how='inner', left_on='intc_date', right_on='STAG_INDUSTRIAL_INC_date')

DDR_CORP = pd.read_csv(path + 'DDR_CORP.csv')
DDR_CORP = DDR_CORP.ix[(DDR_CORP['Date'] >= '2010-01-01') & (DDR_CORP['Date'] <= '2016-12-31'), ['Date', 'Adj_Close']]
DDR_CORP.columns = ['DDR_CORP_date', 'DDR_CORP_close']

total = pd.merge(total, DDR_CORP, how='inner', left_on='intc_date', right_on='DDR_CORP_date')

EQUITY_RESIDENTIAL_PROPERTIES = pd.read_csv(path + 'EQUITY_RESIDENTIAL_PROPERTIES.csv')
EQUITY_RESIDENTIAL_PROPERTIES = EQUITY_RESIDENTIAL_PROPERTIES.ix[(EQUITY_RESIDENTIAL_PROPERTIES['Date'] >= '2010-01-01') & (EQUITY_RESIDENTIAL_PROPERTIES['Date'] <= '2016-12-31'), ['Date', 'Adj_Close']]
EQUITY_RESIDENTIAL_PROPERTIES.columns = ['EQUITY_RESIDENTIAL_PROPERTIES_date', 'EQUITY_RESIDENTIAL_PROPERTIES_close']

total = pd.merge(total, EQUITY_RESIDENTIAL_PROPERTIES, how='inner', left_on='intc_date', right_on='EQUITY_RESIDENTIAL_PROPERTIES_date')

print total.head() #2012-05-10
print total.tail() #2015-01-29

# COLONY_STARWOOD_HOMES = pd.read_csv(path + 'COLONY_STARWOOD_HOMES.csv')
# COLONY_STARWOOD_HOMES = COLONY_STARWOOD_HOMES.ix[(COLONY_STARWOOD_HOMES['Date'] >= '2010-01-01') & (COLONY_STARWOOD_HOMES['Date'] <= '2016-12-31'), ['Date', 'Adj_Close']]
# COLONY_STARWOOD_HOMES.columns = ['COLONY_STARWOOD_HOMES_date', 'COLONY_STARWOOD_HOMES_close']
#
# total = pd.merge(total, COLONY_STARWOOD_HOMES, how='inner', left_on='intc_date', right_on='COLONY_STARWOOD_HOMES_date')

# JPM_P_C = pd.read_csv(path + 'JPM_P_C.csv')
# JPM_P_C = JPM_P_C.ix[(JPM_P_C['Date'] >= '2010-01-01') & (JPM_P_C['Date'] <= '2016-12-31'), ['Date', 'Adj_Close']]
# JPM_P_C.columns = ['JPM_P_C_date', 'JPM_P_C_close']
#
# total = pd.merge(total, JPM_P_C, how='inner', left_on='intc_date', right_on='JPM_P_C_date')
#
# WHA = pd.read_csv(path + 'WHA.csv')
# WHA = WHA.ix[(WHA['Date'] >= '2010-01-01') & (WHA['Date'] <= '2016-12-31'), ['Date', 'Adj_Close']]
# WHA.columns = ['WHA_date', 'WHA_close']
#
# total = pd.merge(total, WHA, how='inner', left_on='intc_date', right_on='WHA_date')

# TII = pd.read_csv(path + 'TII.csv')
# TII = TII.ix[(TII['Date'] >= '2010-01-01') & (TII['Date'] <= '2016-12-31'), ['Date', 'Adj_Close']]
# TII.columns = ['TII_date', 'TII_close']
#
# total = pd.merge(total, TII, how='inner', left_on='intc_date', right_on='TII_date')

data['intc'] = total['intc_close']
data['jpy'] = total['jpy_rate']
data['ms'] = total['ms_close']
data['erh'] = total['erh_close']
data['med'] = total['med_close']
data['barl'] = total['barl_close']
data['STAG_INDUSTRIAL_INC'] = total['STAG_INDUSTRIAL_INC_close']
data['DDR_CORP'] = total['DDR_CORP_close']
data['EQUITY_RESIDENTIAL_PROPERTIES'] = total['EQUITY_RESIDENTIAL_PROPERTIES_close']


# data['JPM_P_C'] = total['JPM_P_C_close']
# data['WHA'] = total['WHA_close']
# data['TII'] = total['TII_close']

data.columns = symbols

print data.head()
# print data.tail()
rets = np.log(data / data.shift(1))
# print rets
# print rets.mean()
# print rets.cov()

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
psharpes = []
for p in range(2500):
    weights = np.random.random(noa)
    weights /= np.sum(weights)
    pweights.append(weights)
    prets.append(np.sum(rets.mean() * weights) * 252)
    pvols.append(np.sqrt(np.dot(weights.T,
                            np.dot(rets.cov() * 252, weights))))
    psharpes.append((np.sum(rets.mean() * weights) * 252)/np.sqrt(np.dot(weights.T,
                            np.dot(rets.cov() * 252, weights))))
prets = np.array(prets)
pvols = np.array(pvols)
psharpes = np.array(psharpes)

# print pweights
# print prets
# print pvols
# print psharpes

df = pd.DataFrame()

for i in range(len(pweights)):
    print i
    # print pweights[i]
    # print pweights[i][0]
    temp = pd.DataFrame({'INTC': pd.Series([pweights[i][0]]), 'JPY': pd.Series([pweights[i][1]]), 'MS': pd.Series([pweights[i][2]]),
                         'ERH': pd.Series([pweights[i][3]]), 'MED': pd.Series([pweights[i][4]]), 'BARL': pd.Series([pweights[i][5]]),
                         'STAG_INDUSTRIAL_INC': pd.Series([pweights[i][6]]), 'DDR_CORP': pd.Series([pweights[i][7]]),
                         'EQUITY_RESIDENTIAL_PROPERTIES': pd.Series([pweights[i][8]])})

    temp['returns'] = prets[i]
    temp['variance'] = pvols[i]
    temp['psharpe'] = psharpes[i]
    temp['invested_amount'] = initial_invest_amt

    INTC_amt = 100000 * pweights[i][0]
    JPY_amt = 100000 * pweights[i][1]
    MS_amt = 100000 * pweights[i][2]
    ERH_amt = 100000 * pweights[i][3]
    MED_amt = 100000 * pweights[i][4]
    BARL_amt = 100000 * pweights[i][5]
    STAG_INDUSTRIAL_INC_amt = 100000 * pweights[i][6]
    DDR_CORP_amt = 100000 * pweights[i][7]
    EQUITY_RESIDENTIAL_PROPERTIES_amt = 100000 * pweights[i][8]

    Calculated_INTC_amt = float(total.ix[(total['intc_date'] == '2015-01-29'), 'intc_close']) / float(
        total.ix[(total['intc_date'] == '2012-05-10'), 'intc_close'])\
          * INTC_amt

    Calculated_JPY_amt = float(total.ix[(total['intc_date'] == '2015-01-29'), 'jpy_rate']) / float(total.ix[
        (total['intc_date'] == '2012-05-10'), 'jpy_rate']) \
                          * JPY_amt

    Calculated_MS_amt = float(total.ix[(total['intc_date'] == '2015-01-29'), 'ms_close']) / float(total.ix[
        (total['intc_date'] == '2012-05-10'), 'ms_close']) \
                         * MS_amt

    Calculated_ERH_amt = float(total.ix[(total['intc_date'] == '2015-01-29'), 'erh_close']) / float(total.ix[
        (total['intc_date'] == '2012-05-10'), 'erh_close']) \
                         * ERH_amt

    Calculated_MED_amt = float(total.ix[(total['intc_date'] == '2015-01-29'), 'med_close']) / float(total.ix[
        (total['intc_date'] == '2012-05-10'), 'med_close']) \
                         * MED_amt

    Calculated_BARL_amt = float(total.ix[(total['intc_date'] == '2015-01-29'), 'barl_close']) / float(total.ix[
        (total['intc_date'] == '2012-05-10'), 'barl_close']) \
                         * BARL_amt

    Calculated_STAG_INDUSTRIAL_INC_amt = float(total.ix[(total['intc_date'] == '2015-01-29'), 'barl_close']) / float(total.ix[
        (total['intc_date'] == '2012-05-10'), 'STAG_INDUSTRIAL_INC_close']) \
                         * STAG_INDUSTRIAL_INC_amt

    Calculated_DDR_CORP_amt = float(total.ix[(total['intc_date'] == '2015-01-29'), 'DDR_CORP_close']) / float(total.ix[
        (total['intc_date'] == '2012-05-10'), 'DDR_CORP_close']) \
                                         * DDR_CORP_amt

    Calculated_EQUITY_RESIDENTIAL_PROPERTIES_amt = float(total.ix[(total['intc_date'] == '2015-01-29'), 'EQUITY_RESIDENTIAL_PROPERTIES_close']) / \
                                                   float(total.ix[(total['intc_date'] == '2012-05-10'), 'EQUITY_RESIDENTIAL_PROPERTIES_close']) \
                                         * EQUITY_RESIDENTIAL_PROPERTIES_amt

    temp['final_amount'] =  Calculated_INTC_amt + Calculated_JPY_amt + Calculated_MS_amt + Calculated_ERH_amt + Calculated_MED_amt + Calculated_BARL_amt \
          + Calculated_STAG_INDUSTRIAL_INC_amt + Calculated_DDR_CORP_amt + EQUITY_RESIDENTIAL_PROPERTIES_amt

    df = df.append(temp)

print df.head()
df.to_csv('asset_distribution.csv', index=False)

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

# print
# print optv
# print
print 'best optimised weight'
optimised_portfolio = optv['x'].round(3)
print optimised_portfolio
print 'optimised returns: ', statistics(optv['x'])[0]
print 'optimised variance: ', statistics(optv['x'])[1]
print 'optimised sharpe: ', statistics(optv['x'])[2]

temp = pd.DataFrame(
    {'INTC': pd.Series([optimised_portfolio[0]]), 'JPY': pd.Series([optimised_portfolio[1]]), 'MS': pd.Series([optimised_portfolio[2]]),
     'ERH': pd.Series([optimised_portfolio[3]]), 'MED': pd.Series([optimised_portfolio[4]]), 'BARL': pd.Series([optimised_portfolio[5]]),
     'STAG_INDUSTRIAL_INC': pd.Series([optimised_portfolio[6]]), 'DDR_CORP': pd.Series([optimised_portfolio[7]]),
     'EQUITY_RESIDENTIAL_PROPERTIES': pd.Series([optimised_portfolio[8]])})
temp['returns'] = statistics(optv['x'])[0]
temp['variance'] = statistics(optv['x'])[1]
temp['psharpe'] = statistics(optv['x'])[2]
temp['invested_amount'] = initial_invest_amt

INTC_amt = 100000 * optimised_portfolio[0]
JPY_amt = 100000 * optimised_portfolio[1]
MS_amt = 100000 * optimised_portfolio[2]
ERH_amt = 100000 * optimised_portfolio[3]
MED_amt = 100000 * optimised_portfolio[4]
BARL_amt = 100000 * optimised_portfolio[5]
STAG_INDUSTRIAL_INC_amt = 100000 * optimised_portfolio[6]
DDR_CORP_amt = 100000 * optimised_portfolio[7]
EQUITY_RESIDENTIAL_PROPERTIES_amt = 100000 * optimised_portfolio[8]

Calculated_INTC_amt = float(total.ix[(total['intc_date'] == '2015-01-29'), 'intc_close']) / float(
    total.ix[(total['intc_date'] == '2012-05-10'), 'intc_close']) \
                      * INTC_amt

Calculated_JPY_amt = float(total.ix[(total['intc_date'] == '2015-01-29'), 'jpy_rate']) / float(
    total.ix[(total['intc_date'] == '2012-05-10'), 'jpy_rate']) \
                     * JPY_amt

Calculated_MS_amt = float(total.ix[(total['intc_date'] == '2015-01-29'), 'ms_close']) / float(
    total.ix[(total['intc_date'] == '2012-05-10'), 'ms_close']) \
                    * MS_amt

Calculated_ERH_amt = float(total.ix[(total['intc_date'] == '2015-01-29'), 'erh_close']) / float(
    total.ix[(total['intc_date'] == '2012-05-10'), 'erh_close']) \
                     * ERH_amt

Calculated_MED_amt = float(total.ix[(total['intc_date'] == '2015-01-29'), 'med_close']) / float(
    total.ix[(total['intc_date'] == '2012-05-10'), 'med_close']) \
                     * MED_amt

Calculated_BARL_amt = float(total.ix[(total['intc_date'] == '2015-01-29'), 'barl_close']) / float(
    total.ix[(total['intc_date'] == '2012-05-10'), 'barl_close']) \
                      * BARL_amt

Calculated_STAG_INDUSTRIAL_INC_amt = float(total.ix[(total['intc_date'] == '2015-01-29'), 'barl_close']) / float(
    total.ix[(total['intc_date'] == '2012-05-10'), 'STAG_INDUSTRIAL_INC_close']) \
                                     * STAG_INDUSTRIAL_INC_amt

Calculated_DDR_CORP_amt = float(total.ix[(total['intc_date'] == '2015-01-29'), 'DDR_CORP_close']) / \
                          float(total.ix[(total['intc_date'] == '2012-05-10'), 'DDR_CORP_close']) \
                          * DDR_CORP_amt

Calculated_EQUITY_RESIDENTIAL_PROPERTIES_amt = float(total.ix[(total['intc_date'] == '2015-01-29'), 'EQUITY_RESIDENTIAL_PROPERTIES_close']) / \
                                               float(total.ix[(total['intc_date'] == '2012-05-10'), 'EQUITY_RESIDENTIAL_PROPERTIES_close']) \
                                               * EQUITY_RESIDENTIAL_PROPERTIES_amt

temp['final_amount'] = Calculated_INTC_amt + Calculated_JPY_amt + Calculated_MS_amt + Calculated_ERH_amt \
                       + Calculated_MED_amt + Calculated_BARL_amt + Calculated_STAG_INDUSTRIAL_INC_amt \
                       + Calculated_DDR_CORP_amt + EQUITY_RESIDENTIAL_PROPERTIES_amt


print temp.head()
temp.to_csv('optimised_asset_allocation.csv', index=False)

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




