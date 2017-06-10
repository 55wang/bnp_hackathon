import quandl
import pandas_datareader.data as web
from datetime import datetime
import pandas_datareader as pdr

quandl.ApiConfig.api_key = "wh4e1aGKQwZyE4RXWP7s"

INTC = quandl.get("EOD/INTC")
INTC.reset_index(level=0, inplace=True)
INTC['bid'] = '35.72 x 1000'
INTC['ask'] = '36.05 x 600'
INTC.to_csv('intc.csv', index=False)
print INTC.head()

jpy = quandl.get("CUR/JPY")
jpy.reset_index(level=0, inplace=True)
jpy['bid'] = 0.009
jpy['ask'] = 0.009
jpy.to_csv('jpy.csv', index=False)
print jpy.head()

# data = quandl.get("CUR/SGD")
# data.reset_index(level=0, inplace=True)
# data['bid'] = '1.384'
# data['ask'] = '1.386'
# data.to_csv('sgd.csv', index=False)
# print data.head()

data = quandl.get("EOD/MS")
data.reset_index(level=0, inplace=True)
data['bid'] = '0.00 x'
data['ask'] = '0.00 x'
data.to_csv('MS.csv', index=False)
print data.head()

data = quandl.get("EOD/ERH")
data.reset_index(level=0, inplace=True)
data['bid'] = '0.00 x'
data['ask'] = '0.00 x'
data.to_csv('ERH.csv', index=False)
print data.head()

data = quandl.get("EOD/MED")
data.reset_index(level=0, inplace=True)
data['bid'] = '0.00 x'
data['ask'] = '0.00 x'
data.to_csv('MED.csv', index=False)
print data.head()




















