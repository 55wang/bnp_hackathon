__author__ = 'luoyuan'

import pickle
import pandas as pd
import os
import numpy as np
import sys
from datetime import datetime, date, time, timedelta

input_paths = os.listdir('./')
for path in input_paths:
    print path
    if 'L3_Operations_24' in path:
        df_L3_Operations_24 = pd.read_pickle(path)

def get_market_impact_loss_FX(today_date, df_excution, df_market_path, instru_iIDValeurs):
    '''
    :return Returns the POST cost for currency instruments.
            It takes into account nuances of matket price for currency.
    :param df_excution:
    :param df_market_path:
    :param instru_iIDValeurs:
    :return:
    '''

    today_date = today_date - timedelta(days=1)
    df_excution = df_excution[df_excution['fQuantity']!=0]
    df_excution = df_excution.ix[1:]
    df_excution = df_excution[df_excution['iIDValeurs']==instru_iIDValeurs]
    print df_excution.head(10)

    value = df_excution['fCostBase'].values #base
    first_volume = df_excution['fQuantity'].values

    df_excution['excution_price'] = np.array(value).astype(float)/first_volume
    first_excution_prices = df_excution['excution_price'].values


    time_series = df_excution['dDateOper']

    df_market = pd.read_csv(df_market_path)
    length = len(time_series)
    # print df_market.head(10)

    # Capture the time for first trade of this instrument.

    if time_series[time_series <= today_date].shape[0] != 0:
        time_series = time_series[time_series <= today_date]
        time = (str(time_series.max()).split()[0])
        print time
    else:
        return 0

    market_prices = []
    times = []
    excution_prices = []
    volume = []
    i = 0

   # df = df.drop('column_name', 1)

    df_mar = df_market[df_market['DATE']==time] #DATE

    if df_mar.shape[0] == 0:
        return 0

    price = df_mar['RATE'].values  #'RATE'
    if len(price)>0:
        for price in price:
            market_prices.append(price)
            times.append(time)
            excution_prices.append(first_excution_prices[i])
            volume.append(first_volume[i])
    i += 1

    price_dif = np.array(market_prices)-np.array(excution_prices)
    # print market_prices
    # print excution_prices

    market_impact_loss =  price_dif*volume
    return market_impact_loss

    final_loss = []
    final_t_series = []
    for i in xrange(len(market_impact_loss)):
        if market_impact_loss[i]<0:
            final_loss.append(market_impact_loss[i])
            final_t_series.append(times[i])

    print volume
    loss_time_series = zip(final_loss, final_t_series)
    print loss_time_series


 #   return loss_time_series


def get_market_impact_loss_stock(today_date, df_excution, df_market_path, instru_iIDValeurs):
    today_date = today_date - timedelta(days=1)

    df_excution = df_excution[df_excution['fQuantity']!=0]
    df_excution = df_excution.ix[1:]
    df_excution = df_excution[df_excution['iIDValeurs']==instru_iIDValeurs]

    print df_excution.head(10)

    value = df_excution['fCostLocal'].values #base
    first_volume = df_excution['fQuantity'].values

    df_excution['excution_price'] = np.array(value).astype(float)/first_volume
    first_excution_prices = df_excution['excution_price'].values

    time_series = df_excution['dDateOper'].copy()
    df_market = pd.read_csv(df_market_path)
    length = len(time_series)
    # print df_market.head(10)

    if time_series[time_series <= today_date].shape[0] != 0:
        time_series = time_series[time_series <= today_date]
        time = (str(time_series.max()).split()[0])
        print time
    else:
        return 0

    market_prices = []
    times = []
    excution_prices = []
    volume = []
    i = 0

    df_mar = df_market[df_market['Date']==time] #DATE
    if df_mar.shape[0] == 0:
        return 0

    price = df_mar['Low'].values  #'RATE'
    if len(price)>0:
        for price in price:
            market_prices.append(price)
            times.append(time)
            excution_prices.append(first_excution_prices[i])
            volume.append(first_volume[i])
    i += 1

    price_dif = np.array(market_prices)-np.array(excution_prices)

#    print market_prices
#    print excution_prices

    market_impact_loss =  price_dif*volume
    return market_impact_loss

    final_loss = []
    final_t_series = []
    final_volume = []
    final_marcket_price = []
    final_excution_price = []
    for i in xrange(len(market_impact_loss)):
        if market_impact_loss[i]<0:
            final_loss.append(market_impact_loss[i])
            final_t_series.append(times[i])
            final_volume.append(volume[i])
            final_marcket_price.append(market_prices[i])
            final_excution_price.append(excution_prices[i])

    print final_volume
    loss_time_series = zip(final_loss, final_t_series)
    print loss_time_series
    #return loss_time_series

    df = pd.DataFrame(data=final_volume)
    print df
    df['times'] = final_t_series
    df['volumn'] = final_volume
    df['excution_price'] =final_excution_price
    df['market_price'] = final_marcket_price
    df['post_cost'] = final_loss

    df = df.drop(0, 1)
    df.to_csv('548.csv', index=False)
    df.to_csv('./post_cost_')

    per_volumn_cost = sum(final_loss)/sum(first_volume)
    print 'per_volumn_cost', per_volumn_cost


today_date = datetime.strptime("2016-10-6", "%Y-%m-%d")
print today_date
# loss_time_series = get_market_impact_loss_stock(today_date, df_L3_Operations_24, df_market_path='./csv/548.csv',
#                                           instru_iIDValeurs=548)
# print loss_time_series

market_impact_loss = get_market_impact_loss_FX(today_date, df_L3_Operations_24, df_market_path='./csv/jpy.csv',
                                               instru_iIDValeurs=159)
print market_impact_loss