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
import altair as alt
import matplotlib.pyplot as plt
import pandas_ta as ta
from forex import extract_fx_data, read_rf, tech_calculator, prophet_analysis, fx_trade_info, fx_correlations

st.set_page_config(page_title= "ForEx Information", page_icon=":currency_exchange:", layout="wide")
# Main function
def main(): 
    st.title('FOREX')
    ticker_df = ['CADUSD=X', 'JPYUSD=X', 'AUDUSD=X', 'EURUSD=X', 'GBPUSD=X', 'CHFUSD=X', 'SGDUSD=X', 'CNYUSD=X', 'ZARUSD=X', 'INRUSD=X', 'BTC-USD', 'ETH-USD', 'USDT-USD', 'DOGE-USD', 'TRX-USD']
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime('2010-01-01'))
    end_date = st.sidebar.date_input("Start Date", value='today', format = "YYYY-MM-DD" )
    fast_window = 4
    slow_window = 11

    one_yrs_ago =(dt.date.today() - pd.DateOffset(years=1)).date().strftime('%Y-%m-%d')
    three_yrs_ago = (dt.date.today() - pd.DateOffset(years=3)).date().strftime('%Y-%m-%d')
    five_yrs_ago = (dt.date.today() - pd.DateOffset(years=5)).date().strftime('%Y-%m-%d')

    fx_risk_info = pd.DataFrame(columns = ticker_df,index =['Sigma','Sharpe', 'Sharpe_1Y', 'Sharpe_5Y'])
    fx_tech_info = pd.DataFrame()
    fx_change = pd.DataFrame(columns = ticker_df,index =['Max_Pips_$','Max_1Y_Pips_$', 'Average_Pips_$','Average_1Y_Pips_$', 'Days_1Y_>20Pips_$','Average_Daily_Return_1Y%', 'Max_Daily_Return_1Y%', 'Min_Daily_Return_1Y%','Days_Daily_Return_1Y_>0.05%','Days_Daily_Return_1Y_<-0.05%','Average_Daily_Return%', 'Max_Daily_Return%', 'Min_Daily_Return%',"Average_Simple_Return%"])
    client_info = pd.read_csv("./Resources/client_info.csv",index_col='name')
    country = client_info['country'][0]
    # st.write(country)

    st.write('Please Select Your Interested FX Pair for Trading:')
    selected_forex = st.selectbox('FX', options=ticker_df, key='forex_selection')

    rf_us = read_rf('us', start_date, end_date)
    st.sidebar.write('USA Risk Free Rate(%) as of ')
    st.sidebar.write(rf_us.iloc[-1])
    grid = st.columns(2)
    with grid[0]:
        st.write('Technical Analysis Results For Your FX Pair')
        df = extract_fx_data(selected_forex, start_date, end_date)
        st.write('Select Window Size for Trading Strategy Optimization')
        fast_window = st.number_input('Fast Window Size:', step=1, format='%d')
        slow_window = st.number_input('Slow Window Size:', step=1, format='%d')
        if st.button('Set Window'):
            fx_tech_info = tech_calculator(df, fast_window, slow_window).dropna()      
            data = fx_tech_info[['Close', 'ema_fast', 'ema_slow']][one_yrs_ago:]
            data['Signal'] = 0.0
            data['Signal']= np.where(data['ema_fast'] > data['ema_slow'], 1.0, 0.0)
            data['Positions'] = data['Signal'].diff()

            st.write('Selected FX Pair '+selected_forex+' Price with EMA for Signals')
            fig1 = plt.figure(figsize = (9, 6))
            plt.plot(data.index, fx_tech_info[['Close', 'ema_fast', 'ema_slow']][one_yrs_ago:])
            plt.plot(data[data['Positions'] == 1].index, data['ema_fast'][data['Positions'] == 1], marker = '^', linewidth = 0.01, color='black')
            plt.plot(data[data['Positions'] == -1].index, data['ema_fast'][data['Positions'] == -1], marker = 'v',linewidth = 0.01, color='red')
            days = len(data.index)
            plt.xticks(ticks=data.index[0:(days-1):30],labels=data.index[0:(days-1):30],rotation=70)
            fig1.tight_layout()
            st.pyplot(fig1)

            st.write('Selected FX Pair '+selected_forex+' Price with RSI for Signals')
            overbought = st.number_input('Overbought RSI Value:', value = 50, step=1, format='%d')
            oversold = st.number_input('Oversold RSI Value:', value = 40, step=1, format='%d')
            
            data_rsi = fx_tech_info[['rsi_14']][three_yrs_ago:]
            
            scale = fx_tech_info['rsi_14'][three_yrs_ago:].mean()/(fx_tech_info['Close'][three_yrs_ago:].mean())
            data2 = fx_tech_info[['Close']][three_yrs_ago:]*scale
            data2.columns = ['Price_Scaled']
            data2['rsi_14'] = fx_tech_info['rsi_14'][three_yrs_ago:]
            data_rsi['Signal'] = 0
            fig2 = plt.figure(figsize = (9, 6))
            data_rsi['Signal'][data_rsi['rsi_14'] > overbought] = -1
            data_rsi['Signal'][data_rsi['rsi_14'] < oversold] = 1
            data_rsi['Positions'] = data_rsi['Signal'].diff()
                
            data2['overbought'] = overbought
            data2['oversold'] = oversold
            st.line_chart(data2,use_container_width=True)
            fig2.tight_layout()
                
                
                

            st.write('Selected FX Pair '+selected_forex+' Price with Bollinger Bands for Signals')
            fig3 = plt.figure(figsize = (6, 6))
            st.line_chart(fx_tech_info[['Close','BB_lower','BB_middle','BB_upper']][one_yrs_ago:],use_container_width=True)
            fig3.tight_layout()
        

    with grid[1]:

        st.write(selected_forex+' Prophet Analysis and Predictions')
        forecast_days=st.number_input('Please Specify FX Price Prophet Projection Days', value=None, step=1, format='%d')
        if st.button('Predict'):
            prophet_analysis(selected_forex, start_date, end_date, forecast_days)
        
        st.write("USA Risk Free Rate Movement")
        st.line_chart(rf_us,use_container_width=True)
        
        if selected_forex == 'CADUSD=X':
            st.sidebar.write('CA Risk Free Rate(%) as of ')
            rf_ca = read_rf('ca', start_date, end_date)
            st.sidebar.write(rf_ca.iloc[-1])
            st.write("CA Risk Free Rate Movement")
            st.line_chart(rf_ca,use_container_width=True )



# def extract_fx_data(forex, start_date):
#     forex_list = ticker_df
#     forex_name = st.sidebar.selectbox('ForEX', options = ticker_df) #'Enter Stock_Name from the above displayed list for Analysis:')
#     start_date = pd.to_datetime(st.date_input("State Date", start_date)) #'Enter Start Date for Historical Data in yyyy-mm-dd:')
#     today = datetime.today().strftime('%Y-%m-%d')    
#     forex_df = yf.download(forex,start_date,today)
#     forex_df.index = pd.to_datetime(forex_df.index).strftime('%Y-%m-%d')
#     forex_df = pd.DataFrame(forex_df)
#     return(forex_df)

# st.write(forex_df)

# @st.cache_data

# def main():
#     st.title('FOREX')
#     #option_1 = 

    
   


if __name__ == "__main__":
    main()
    