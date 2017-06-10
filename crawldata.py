import quandl
import pandas_datareader.data as web
from datetime import datetime
import pandas_datareader as pdr

quandl.ApiConfig.api_key = "wh4e1aGKQwZyE4RXWP7s"

INTC = quandl.get("EOD/INTC")
INTC.reset_index(level=0, inplace=True)
print INTC.head()
INTC.to_csv('intc.csv', index=False)

jpy = quandl.get("CUR/JPY")
jpy.reset_index(level=0, inplace=True)
jpy.to_csv('jpy.csv', index=False)
print jpy




