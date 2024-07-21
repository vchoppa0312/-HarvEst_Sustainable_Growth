import streamlit as st
from dataclasses import dataclass
from typing import Any, List
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import os
import yfinance as yf
import plotly.express as px
from sklearn import linear_model
import warnings
from stocks import snp500_tickers
from stocks import extract_stock_data
from stocks import stock_risk_profiling
from forex import extract_fx_data, read_rf, tech_calculator, prophet_analysis, fx_trade_info, fx_correlations
warnings.filterwarnings('ignore')
start_date = '2010-01-01'

st.set_page_config(page_title= "Investment Information", page_icon=":tada:", layout="wide")
#@st.cache_data
# Main function
def main():
    client_info = pd.read_csv("./Resources/client_info.csv",index_col='name')
    investmentTerm = client_info['investmentTerm'][0]
    investment_amount = client_info['investableCashAsset'][0]
    clientRiskAppetite= client_info['clientRiskAppetite'][0]

    if investmentTerm =='Short (<1 year)' and clientRiskAppetite=='Very High' or 'High':
        investmentTermDays = 21*6
    elif investmentTerm =='Short (<1 year)'and clientRiskAppetite =='Balanced' or 'Conservative':
        investmentTermDays = 210
    elif investmentTerm =='Medium (1-3 years)'and clientRiskAppetite =='Very High':
        investmentTermDays = 252
    elif investmentTerm =='Medium (1-3 years)'and clientRiskAppetite =='High' or 'Balanced':
        investmentTermDays = 2*252
    elif investmentTerm =='Medium (1-3 years)'and clientRiskAppetite =='Conservative':
        investmentTermDays = 3*252
    elif investmentTerm =='Long (>3 years)'and clientRiskAppetite =='Very High' or 'High':
        investmentTermDays = 3*252
    elif investmentTerm =='Long (>3 years)'and clientRiskAppetite =='Balanced':
        investmentTermDays= 5*252
    elif investmentTerm =='Long (>3 years)'and clientRiskAppetite =='Conservative':
        investmentTermDays = 7*252

    roi_df = pd.read_csv('./Resources/roi.csv')
    target_roi = roi_df.iloc[0,1]/100

    ticker_dict = snp500_tickers()
    stock_dict={}   
    st.title("Stocks Investment Options & Recommendations") 
    # st.header('Recommendations')
    st.sidebar.header("Select a stock to analyze its Risk Profile ")
    stock = st.sidebar.selectbox("Stock Selection ",ticker_dict,placeholder="Select a stock")

    stock_dict[stock] = ticker_dict[stock]
    stocks_close = extract_stock_data(stock_dict)
    df = stocks_close.reset_index()
        
    tab1, tab2, tab3, tab4 = st.tabs(['Stocks','ETF', 'Forex Exchange','Project Finance'])
    #start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime('2020-01-01'))
    #start_date = '2010-01-01'
    
    data,analysis,profile,prediction = stock_risk_profiling(investmentTermDays)
    
        #if st.sidebar.button("Select"):
    with tab1:
            #st.write('display stock information')
            col1,col2,col3,col4=st.columns(4)
            with col1:
                st.write("Stock Selected: ",stock)
            with col2:
                 st.write(f"Investment Amount: ${investment_amount}")
            with col3:
                 st.write("Investment Period: ",investmentTerm)
            with col4:
                 st.write(f"Min Targte RoI: {round(target_roi*100,2)}%")
            table = st.columns(2)
            with table[0]:
                st.write("###### Stock Details")
                st.dataframe(pd.DataFrame(data.loc[stock]),width=350)
                st.write("###### Risk Profile - Compared with Pool of S&P 500 Stocks")
                st.dataframe(pd.DataFrame(profile.loc[stock]),width=350)
            with table[1]:
                st.write("###### Key Numbers")
                st.dataframe(pd.DataFrame(analysis.loc[stock]),width=350)
                st.write("###### Some Predictions")
                st.dataframe(pd.DataFrame(prediction.loc[stock]),width=350)
            st.markdown("#### Scroll down for Stock Price & Return Movements")
            st.write("##### Last Year Price Movement with CandleStick Depiction")
            fig = go.Figure(data=[go.Candlestick(x=df.iloc[-252:,0],
                                                        open=df.iloc[-252:,1],
                                                        high=df.iloc[-252:,2],
                                                        low=df.iloc[-252:,3],
                                                        close=df.iloc[-252:,4])])
            st.plotly_chart(fig, use_container_width=True) 
            
            grid = st.columns(3)
            with grid[0]:
                    data = stocks_close.iloc[-21:][f"{stock}_Close"]
                    st.write("##### Last 1Month Daily Log Returns Movement")
                    log_data = np.log(data/data.shift(1))
                    st.area_chart(log_data,use_container_width=True, color="#ffaa00")    
            with grid[1]:
                    data = stocks_close.iloc[-252:][f"{stock}_Close"]
                    st.write("##### Last 1Year Daily Mean Returns Movement")
                    mean_data = data.pct_change()
                    st.line_chart(mean_data,use_container_width=True, color="#8284A4")
            with grid[2]:
                    st.write("##### Last 5Years Daily Price Movement")
                    data_5= stocks_close.iloc[-252*5:][f"{stock}_Close"]
                    st.line_chart(data_5,use_container_width=True, color="#79BDA9")
                    

            if st.button('Build A Portfolio'):
                st.switch_page("pages/4_Client_Portfolio.py")
        #     st.write('insert summary')
    st.write("***Data Source: Yahoo Finance***")

    # if st.button('Back'):
    #      st.switch_page("pages/2_Investment_Info.py")
         
            

         

    with tab2:
        st.header("ETF")
        st.write("COMING SOON! :seedling:")

    with tab3:
        st.header("Forex Exchange")
        ticker_df = ['CADUSD=X', 'JPYUSD=X', 'AUDUSD=X', 'EURUSD=X', 'GBPUSD=X', 'CHFUSD=X', 'SGDUSD=X', 'CNYUSD=X', 'ZARUSD=X', 'INRUSD=X', 'BTC-USD', 'ETH-USD', 'USDT-USD', 'DOGE-USD', 'TRX-USD']
        etf_df = ['USD', 'FNGU', 'GBTC', 'SMH', 'TQQQ', 'ITB', 'TECL','NVDX', 'FBL', 'TSLZ']
        fast_window = 21
        slow_window = 50
        end_date = st.sidebar.date_input('End Date',value='today')
        rf_us = read_rf('us', start_date, end_date)
        fast_window = 21
        slow_window = 50
        
        fx_change, fx_risk_info = fx_trade_info(ticker_df, start_date, end_date, fast_window, slow_window, rf_us)

        #Select FX pairs that has over 130 days in recnet year over 20Pips change
        tradable_fx=fx_change.columns[fx_change.loc['Days_1Y_>20Pips_$']>130]
        fx_change_new =fx_change[tradable_fx]
        fx_risk_info_new = fx_risk_info[tradable_fx]
        client_info = pd.read_csv("./Resources/client_info.csv",index_col='name')
        investment_amount = client_info['investableCashAsset'][0]
        clientRiskAppetite= client_info['clientRiskAppetite'][0]
        investmentTerm = client_info['investmentTerm'][0]
                        
        risk_select_column = 'Sharpe_1Y'
        if investmentTerm =='Medium (1-3 years)':
            risk_select_column = 'Sharpe_3Y'
        elif investmentTerm == "Long(>3 years)":
            risk_select_column = 'Sharpe_5Y'

        
        very_high_risk_fx =fx_risk_info.columns[fx_risk_info.loc[risk_select_column].abs()<0.5]
        high_risk_fx = fx_risk_info_new.columns[(fx_risk_info_new.loc[risk_select_column].abs()>=0.5)&(fx_risk_info_new.loc[risk_select_column].abs()<1)]
        balanced_risk_fx = fx_risk_info_new.columns[(fx_risk_info_new.loc[risk_select_column].abs()>=1)&(fx_risk_info_new.loc[risk_select_column].abs()<1.5)]
        low_risk_fx = fx_risk_info_new.columns[fx_risk_info_new.loc[risk_select_column].abs()>=1.5]

        if clientRiskAppetite == "Very High":
            st.write('FX Daily Change Info (1Bips = 0.0001$)')
            st.write(fx_change[very_high_risk_fx])
            st.write('FX Risk Info for Very High Risk Selection')
            st.dataframe(fx_risk_info[very_high_risk_fx].sort_values(axis = 1, by = 'Sharpe_1Y', ascending=False))
        elif clientRiskAppetite == "High":
            st.write('FX Daily Change Info (1Bips = 0.0001$)')
            st.write(fx_change_new[high_risk_fx])
            st.write('FX Risk Info for High Risk Selection')
            st.dataframe(fx_risk_info_new[high_risk_fx].sort_values(axis = 1, by = 'Sharpe_1Y', ascending=False))
        elif clientRiskAppetite =='Balanced':
            st.write('FX Daily Change Info (1Bips = 0.0001$)')
            st.write(fx_change_new[balanced_risk_fx])
            st.write('FX Risk Info for High Risk Selection')
            st.dataframe(fx_risk_info_new[balanced_risk_fx].sort_values(axis = 1, by = 'Sharpe_1Y', ascending=False))
        elif clientRiskAppetite =='Conservative':
            st.write('FX Daily Change Info (1Bips = 0.0001$)')
            st.write(fx_change_new[low_risk_fx])
            st.write('FX Risk Info for High Risk Selection')
            st.dataframe(fx_risk_info_new[low_risk_fx].sort_values(axis = 1, by = 'Sharpe_1Y', ascending=False))
        else:
            st.write('Please Select your Risk Appetite on Client Information Page.')

        fx_correlations(ticker_df, start_date, end_date)

        


        st.write('Please Select Your Interested FX Pair for Trading:')
        selected_forex = st.selectbox('FX', options=ticker_df, key='forex_selection')
        if selected_forex == 'CADUSD=X':
            st.sidebar.write('CA Risk Free Rate(%) as of ')
            rf_ca = read_rf('ca', start_date, end_date)
            st.sidebar.write(rf_ca.iloc[-1])

        df = extract_fx_data(selected_forex, start_date, end_date)
        fx_tech_info = tech_calculator(df, fast_window, slow_window).dropna()  
        st.write('Technical Analysis Results For Your FX Pair')
        st.dataframe(fx_tech_info)

        st.write(selected_forex+' Prophet Analysis and Predictions')
        forecast_days=st.number_input('Please Specify FX Price Prophet Projection Days', value=None, step=1, format='%d')
        if forecast_days:
             prophet_analysis(selected_forex, forecast_days)



    with tab4:
        st.header("Project Finance")
        st.write("COMING SOON! :seedling:")

    if st.button('Previous Page'):
         st.switch_page("pages/2_Investment_Info.py")
if __name__ == "__main__":
    main()
