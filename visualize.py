import pickle
import pandas as pd
from datetime import datetime
import numpy as np
import random
import sys

symbols = ['intc', 'jpy', 'ms', 'erh', 'med', 'barl', 'STAG_INDUSTRIAL_INC', 'DDR_CORP', 'EQUITY_RESIDENTIAL_PROPERTIES']
instrument_value = [548, 159, 292391, 378790, 1139292, 1072180, 1123568, 1140653, 29665]

df = pd.read_pickle('L12_2_Positions_12.pkl')

result = pd.DataFrame()

date = [datetime.strptime('2016-11-30', '%Y-%m-%d'), datetime.strptime('2016-12-31', '%Y-%m-%d'), datetime.strptime('2017-01-31', '%Y-%m-%d'),
        datetime.strptime('2017-02-28', '%Y-%m-%d'), datetime.strptime('2017-03-31', '%Y-%m-%d'),datetime.strptime('2017-04-30', '%Y-%m-%d')]

result['Date'] = date

print result
total_value = 0

for i in instrument_value:
    temp = df[df['iIDValeurs'] == i]
    print temp.shape
    temp = temp[['dDatePosition', 'fValueBase']]
    temp.columns = [str(i)+'dDatePosition', str(i)]
    result = result.merge(temp, how='left', left_on='Date', right_on=str(i)+'dDatePosition')
    result = result.drop(str(i)+'dDatePosition', 1)
    total_value = total_value + float(result[str(i)].sum())

print result.head(20)
print

print total_value



Country = ['SG', 'HK', 'MY']
Sector = ['Tech', 'Financial', 'Private']
prejson = pd.DataFrame()

random.seed(1234)
for name in instrument_value:
    temp = pd.DataFrame([{'Instrument_Name': name, 'Country': Country[random.randint(0, 2)],
                            'Sector': Sector[random.randint(0, 2)], 'Value': result[str(name)].sum(),
                                      'percentage_total': float(result[str(name)].sum()/float(total_value))}])
    prejson = prejson.append(temp)

    # weights


prejson['percentage_total'] = prejson['percentage_total']/np.sum(prejson['percentage_total'])
prejson.columns = prejson.columns.astype(str)

print prejson['percentage_total'].sum()
print prejson.to_json(orient='records', lines=True)
