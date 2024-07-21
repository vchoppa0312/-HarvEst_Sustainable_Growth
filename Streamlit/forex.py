import yfinance as yf
from pandas.tseries.offsets import DateOffset
from datetime import datetime
import datetime as dt
import pandas_ta as ta
import streamlit as st
from dataclasses import dataclass
from typing import Any, List
import numpy as np
import pandas as pd
import seaborn as sns
import holoviews as hv
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pytz
import plotly.express as px
from prophet import Prophet
import matplotlib.pyplot as plt

#ticker_df = ['CADUSD=X', 'JPYUSD=X', 'AUDUSD=X', 'EURUSD=X', 'GBPUSD=X', 'CHFUSD=X', 'SGDUSD=X', 'CNYUSD=X', 'ZARUSD=X', 'INRUSD=X', 'BTC-USD', 'ETH-USD', 'USDT-USD', 'DOGE-USD', 'TRX-USD']
#forex_name = 'CADUSD=X'
#start_date = '2014-01-01'

one_yrs_ago =(dt.date.today() - pd.DateOffset(years=1)).date().strftime('%Y-%m-%d')
three_yrs_ago = (dt.date.today() - pd.DateOffset(years=3)).date().strftime('%Y-%m-%d')
five_yrs_ago = (dt.date.today() - pd.DateOffset(years=5)).date().strftime('%Y-%m-%d')

fx_df = pd.DataFrame()

fx_tech_info = pd.DataFrame()



def extract_fx_data(forex, start_date, end_date):
    #forex_list =ticker_df
    #forex_name = input('Enter Stock_Name from the above displayed list for Analysis:')
    #start_date = input('Enter Start Date for Historical Data in yyyy-mm-dd:')
    if end_date == None:
        end_date= datetime.today().strftime('%Y-%m-%d')    
    forex_df = yf.download(forex,start_date,end_date)
    forex_df.index = pd.to_datetime(forex_df.index).strftime('%Y-%m-%d')
    #st.dataframe(forex_df)
    return(forex_df)

def read_rf(country, start_date, end_date):
    rf = extract_fx_data('^IRX', start_date, end_date)
    #rf.index = pd.to_datetime(rf.index).strftime('%Y-%m-%d')
    rf.drop(columns = ['Open','High','Low','Adj Close','Volume'], inplace = True)
    if country == 'us':
        rf['3M'] = rf['Close']
        rf['10Y'] = extract_fx_data('^TNX', start_date, end_date)['Close']
    elif country == 'ca':
        rf_ca = pd.read_csv ('./Resources/risk_free_ca.csv', index_col='Date', parse_dates=True)
        rf_ca.index = pd.to_datetime(rf_ca.index).strftime('%Y-%m-%d')
        lastrf=rf_ca.loc[rf_ca.index < rf.index[0]]
        rf = rf.join(rf_ca, how='left')
        rf['3M'].iloc[0]=lastrf['3M'].iloc[-1]
        rf['1Y'].iloc[0]=lastrf['1Y'].iloc[-1]
    rf.fillna(method = 'ffill', inplace=True)
    rf.drop(columns = ['Close'], inplace = True)
    return rf


def sharpe_ratio(return_df, N, rf):
    #total average return and sigma
    avg_return = return_df.mean(skipna=True)*N-rf.mean(skipna=True)/100
    avg_sigma = return_df.std(skipna=True)
    #recent 1Y average return and sigma
    n_recentyear = return_df.loc[one_yrs_ago:].count()
    avg_recentyear_return =(return_df.loc[one_yrs_ago:].mean(skipna=True)*n_recentyear-rf.loc[one_yrs_ago:].mean(skipna=True)/100)
    avg_recentyear_sigma = return_df.loc[one_yrs_ago:].std(skipna=True)
    #recent 3Y average return and sigma
    n_3y = return_df.loc[three_yrs_ago:].count()
    avg_3y_return =(return_df.loc[three_yrs_ago:].mean(skipna=True)*n_3y-rf.loc[three_yrs_ago:].mean(skipna=True)/100)
    avg_3y_sigma = return_df.loc[three_yrs_ago:].std(skipna=True)
    #recent 5Y average return and sigma
    n_5y = return_df.loc[five_yrs_ago:].count()
    avg_5y_return =(return_df.loc[five_yrs_ago:].mean(skipna=True)*n_5y-rf.loc[five_yrs_ago:].mean(skipna=True)/100)
    avg_5y_sigma = return_df.loc[five_yrs_ago:].std(skipna=True)

    sharpe_total = avg_return/(avg_sigma*np.sqrt(n_recentyear)*np.sqrt(N))
    sharpe_recentyear = avg_recentyear_return/(avg_recentyear_sigma*np.sqrt(n_recentyear))
    sharpe_3year = avg_3y_return/(avg_3y_sigma*np.sqrt(n_3y))
    sharpe_5year = avg_5y_return/(avg_5y_sigma*np.sqrt(n_5y))
    return (sharpe_total,sharpe_recentyear, sharpe_3year, sharpe_5year)

def tech_calculator(forexdf, fast_window, slow_window):
    tech_df = forexdf
    # Daily Returns
    tech_df['daily_simple_return'] = tech_df['Close'].pct_change()
    tech_df['daily_return'] = np.log(tech_df['Close']).diff()
    # Daily Price change range in bips, 1bips=$0.0001
    tech_df['daily_change_range_pips'] = (tech_df['High'] - tech_df['Low'])*10000
    # 21Day Volatility
    tech_df['volatility_21'] = tech_df['Close'].rolling(window=21, min_periods=1).std()
    # Exponential Moving Avg Fast & Slow Windows
    tech_df['ema_fast'] = tech_df.ta.ema(length=fast_window)
    tech_df['ema_slow'] = tech_df.ta.ema(length=slow_window)
    # RSI Momemtum Indicator 14Days
    tech_df['rsi_14'] = tech_df.ta.rsi()
    # MACD Momentum Indicator
    tech_df[['ema_12', 'ema_26', 'MACD_signal']] = tech_df.ta.macd()
    # Bolinger Bands %B Indicator
    tech_df[['BB_lower','BB_middle','BB_upper','BBP','BB%B']]= tech_df.ta.bbands()
    # Dropping trivial Columns
    tech_df = tech_df.drop(columns = ['Open','High','Low','Adj Close','Volume','BBP'])
    return tech_df

def fx_correlations(ticker_df, start_date, end_date):
    for ticker in ticker_df:
        df = extract_fx_data(ticker, start_date, end_date)
        fx_df[ticker] = df['Close']
    fig1 = plt.figure(figsize = (15, 15))
    sns.heatmap(fx_df.corr(), annot = True, fmt = '.2f', cmap = 'PiYG',
            linewidths = .5, square = True, annot_kws = {'fontsize':8, 'fontweight': 'bold'}, vmin = -1, vmax = 1)
    st.pyplot(fig1)
    return (fig1,fx_df)

def fx_trade_info(ticker_df, start_date, end_date, fast_window, slow_window, rf_us):
    fx_risk_info = pd.DataFrame(columns = ticker_df,index =['Sigma','Sharpe', 'Sharpe_1Y','Sharpe_3Y','Sharpe_5Y'])
    fx_change = pd.DataFrame(columns = ticker_df,index =['Max_Pips_$','Max_1Y_Pips_$', 'Average_Pips_$','Average_1Y_Pips_$', 'Days_1Y_>20Pips_$','Average_Daily_Return_1Y%', 'Max_Daily_Return_1Y%', 'Min_Daily_Return_1Y%','Days_Daily_Return_1Y_>0.05%','Days_Daily_Return_1Y_<-0.05%','Average_Daily_Return%', 'Max_Daily_Return%', 'Min_Daily_Return%',"Average_Simple_Return%"])
    for ticker in ticker_df:
        df = extract_fx_data(ticker, start_date, end_date)
        tech_results = tech_calculator(df, fast_window, slow_window)
        fx_change[ticker]['Max_Pips_$'] = tech_results[['daily_change_range_pips']].max().iloc[0] #minimum is found to be 0
        fx_change[ticker]['Max_1Y_Pips_$'] = tech_results[['daily_change_range_pips']].loc[one_yrs_ago:].max().iloc[0]
        val = tech_results[['daily_change_range_pips']].loc[one_yrs_ago:]
        fx_change[ticker]['Days_1Y_>20Pips_$'] = len(val[val.daily_change_range_pips>20])
        fx_change[ticker]['Average_Pips_$'] = tech_results[['daily_change_range_pips']].mean(skipna=True).iloc[0]
        fx_change[ticker]['Average_1Y_Pips_$'] = tech_results[['daily_change_range_pips']].loc[one_yrs_ago:].mean(skipna=True).iloc[0]
        fx_change[ticker]['Average_Simple_Return%'] = tech_results[['daily_simple_return']].mean(skipna=True).iloc[0]*100
        fx_change[ticker]['Max_Daily_Return%'] = tech_results[['daily_return']].max().iloc[0]*100
        fx_change[ticker]['Min_Daily_Return%'] = tech_results[['daily_return']].min().iloc[0]*100
        fx_change[ticker]['Average_Daily_Return%'] = tech_results[['daily_return']].mean(skipna=True).iloc[0]*100
        fx_change[ticker]['Average_Daily_Return_1Y%'] = tech_results[['daily_return']].loc[one_yrs_ago:].mean(skipna=True).iloc[0]*100
        fx_change[ticker]['Max_Daily_Return_1Y%'] = tech_results[['daily_return']].loc[one_yrs_ago:].max().iloc[0]*100
        fx_change[ticker]['Min_Daily_Return_1Y%'] = tech_results[['daily_return']].loc[one_yrs_ago:].min().iloc[0]*100
        val_return = tech_results[['daily_return']].loc[one_yrs_ago:]*100
        fx_change[ticker]['Days_Daily_Return_1Y_>0.05%']= len(val_return[val_return.daily_return>0.0005])
        fx_change[ticker]['Days_Daily_Return_1Y_<-0.05%']= len(val_return[val_return.daily_return<-0.0005])
        fx_risk_info[ticker]['Sigma']= df['Close'].std(skipna=True)
        sharpe_total,sharpe_recentyear, sharpe_3year, sharpe_5year = sharpe_ratio(tech_results['daily_return'],262, rf_us['3M'])
        fx_risk_info[ticker]['Sharpe']= sharpe_total
        fx_risk_info[ticker]['Sharpe_1Y']= sharpe_recentyear
        fx_risk_info[ticker]['Sharpe_3Y']= sharpe_3year
        fx_risk_info[ticker]['Sharpe_5Y']= sharpe_5year

    return fx_change, fx_risk_info


def prophet_analysis(name, forcast_time):
    hv.extension('bokeh')
    prophet_df=fx_df[[name]].reset_index()
    prophet_df.columns=['ds', 'y']
    m = Prophet(yearly_seasonality=20)
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods= forcast_time, freq = 'D')
    future_prices = m.make_future_dataframe(periods=forcast_time)
    forecast = m.predict(future_prices)
    forecast_band = forecast.iloc[-60:,:]
    fig3 =m.plot_components(forecast_band)
    st.pyplot(fig3)
    fig4 = m.plot(forecast)
    st.pyplot(fig4)
    return (forecast_band, forecast)

# prophet_analysis(forex_name, 60)







# fig1 = plt.figure()
# plt.plot(fx_df)
# plt.xlabel("Date")
# plt.legend(ticker_df, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5)
# st.pyplot(fig1)
# fig2 = plt.figure(figsize = (15, 15))
# sns.heatmap(fx_df.corr(), annot = True, fmt = '.2f', cmap = 'PiYG',
#             linewidths = .5, square = True, annot_kws = {'fontsize':8, 'fontweight': 'bold'}, vmin = -1, vmax = 1)
# st.pyplot(fig2)



