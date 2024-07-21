# Import necessary libraries
import numpy as np
import pandas as pd
import hvplot.pandas
from scipy.stats import norm
import matplotlib.pyplot as plt
import os

import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
from pandas.tseries.offsets import DateOffset
from datetime import datetime
start_date='2010-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')

def snp500_master_data(investment_amount,investment_term_days,target_roi):
    wiki_data = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    ticker_df = wiki_data[0]
    ticker_list= ticker_df['Symbol'].to_list()
    # Correcting Ticker Errors in Wiki_data
    ticker_list[62]='BRK-B' 
    ticker_list[78]='BF-B'
    stock_list = ticker_df['Security'].to_list()
    stock_ticker_list = []
    stock_ticker_dict = {}
    stock_ticker_dict['S&P 500'] = '^GSPC'
    #stock_ticker_list.append(stock_ticker_dict)
    i=0
    for stock in stock_list:
        stock_ticker_dict[stock] = ticker_list[i]
        #stock_ticker_list.append(stock_ticker_dict)
        i+=1
#     return(stock_ticker_dict)
# # Only For Master File Generation
# def stock_master_data():
    ticker_dict = stock_ticker_dict
    stocks_price_data = pd.DataFrame()
    for stock,ticker in ticker_dict.items():
        stock_df = yf.download(ticker,start_date,end_date)
        stock_df.index = pd.to_datetime(stock_df.index)
        stock_df = stock_df[['Close']]
        stock_df = stock_df.rename(columns={'Close':stock})
        stocks_price_data = pd.concat([stocks_price_data, stock_df],axis=1)
    stocks_price_data.to_csv('./Resources/stocks_price_data.csv',index=True)
#     return(stocks_price_data)

# # # Extract Stock Price for Select Stock:Ticker Pair
# # def extract_stock_data(ticker_dict):
# #     # Input dictionary with {stock_name:ticker_symbol}
# #     stocks_data = pd.DataFrame()
# #     for stock,ticker in ticker_dict.items():
# #         stock_df = yf.download(ticker,start_date,end_date)
# #         stock_df.index = pd.to_datetime(stock_df.index)
# #         stock_df = stock_df[['Close']]
# #         stock_df = stock_df.rename(columns={'Close':stock})
# #         stocks_data = pd.concat([stocks_data, stock_df],axis=1)

# #     return(stocks_data)

# # Extract Stock Fundamentals Select Stock:Ticker Pair
# def stock_key_financials():
#     ticker_dict = snp500_tickers()
    stock_fundamentals = pd.DataFrame(index = list(ticker_dict))
    stock_fundamentals.index.name = 'Stock'
    for stock,ticker in ticker_dict.items():
        key_info = yf.Ticker(ticker).info
        stock_fundamentals.loc[stock,'Ticker'] = ticker
        try:
            stock_fundamentals.loc[stock,'Industry'] = key_info['industry']
        except KeyError:
            pass
        try:
            stock_fundamentals.loc[stock,'Sector'] = key_info['sector']
        except KeyError:
            pass
        try:
            stock_fundamentals.loc[stock,'MarketCap(BN)'] = round(key_info['marketCap']/1000000000,2)
        except KeyError:
            pass
        try:
            stock_fundamentals.loc[stock,'SMA50'] = key_info['fiftyDayAverage']
        except KeyError:
            pass
        try:
            stock_fundamentals.loc[stock,'SMA200'] = key_info['twoHundredDayAverage']
        except KeyError:
            pass
        try:
            stock_fundamentals.loc[stock,'ProfitMargins'] = key_info['profitMargins']
        except KeyError:
            pass
        try:
            stock_fundamentals.loc[stock,'Beta'] = key_info['beta']   
        except KeyError:
            pass
        try:
            stock_fundamentals.loc[stock,'DividendYield'] = key_info['dividendYield']
        except KeyError:
            pass
        try:
            stock_fundamentals.loc[stock,'P/B_Ratio'] = key_info['priceToBook']
        except KeyError:
            pass
        try:
            stock_fundamentals.loc[stock,'PEG_Ratio'] = key_info['pegRatio']
        except KeyError:
            pass
        try:
            stock_fundamentals.loc[stock,'Debt/Equity'] = key_info['debtToEquity']
        except KeyError:
            pass
        try:
            stock_fundamentals.loc[stock,'EPS'] = key_info['trailingEps']
        except KeyError:
            pass
        try:
            stock_fundamentals.loc[stock,'NoOfOpinions'] = key_info['numberOfAnalystOpinions']
        except KeyError:
            pass
        try:
            stock_fundamentals.loc[stock,'Recommendation'] = key_info['recommendationKey']
        except KeyError:
            pass
        stock_fundamentals.loc[stock_fundamentals.index=='S&P 500','Beta'] = 1.00

        stock_fundamentals.to_csv('./Resources/stock_fundamentals_data.csv',index=True)
#     return(stock_fundamentals)

# # Extracting Risk Free Rates
# def risk_free_rates():
    rf_dict = {'13W_Risk_Free':'^IRX','10Y_Risk_Free':'^TNX'}
    rf_rates = pd.DataFrame()
    for key, value in rf_dict.items():
        rf_rates_df = yf.download(value,start_date,end_date)
        rf_rates_df.index = pd.to_datetime(rf_rates_df.index)
        rf_rates[key] = rf_rates_df['Close']

#     return(rf_rates)

# # Sharpe Ratio
# def stock_sharpe_ratio(investment_term_days):
    stocks_data = stocks_price_data
    #stocks_data = stock_price_data()
    rf_rate = rf_rates
    if investment_term_days<= 252:
        rate = '13W_Risk_Free'
    else:
        rate = '10Y_Risk_Free'

    Sharpe_Ratio = pd.DataFrame(index=stocks_data.columns)
    Sharpe_Ratio.index.name = 'Stock'

    log_returns = np.log(stocks_data/stocks_data.shift(1))
    rolling_mean = log_returns.rolling(window=investment_term_days).mean()
    returns = np.exp(rolling_mean*investment_term_days)-1
    avg_returns = returns.mean(skipna=True)
    avg_volatility = returns.std(skipna=True)
    avg_rf_rate = rf_rate[rate]*(investment_term_days/252)/100
    sharpe_ratio = (avg_returns-avg_rf_rate.mean())/avg_volatility
    Sharpe_Ratio['avg_returns']= avg_returns
    Sharpe_Ratio['avg_volatility']=avg_volatility
    Sharpe_Ratio['Avg_Annual_Returns']= (np.exp(rolling_mean*252)-1).mean(skipna=True)
    Sharpe_Ratio['Avg_Annual_Volatility']=(np.exp(rolling_mean*252)-1).std(skipna=True)
    Sharpe_Ratio['Sharpe_Ratio']=sharpe_ratio

    Sharpe_Ratio.to_csv('./Resources/stock_sharpe_ratio.csv',index=True)

#     return(Sharpe_Ratio)

# #MCSimulation of Stock Returns
# def stock_mc_simulation(investment_term_days,target_roi):
    stocks = stocks_price_data
    #stocks_data = stock_price_data()
    log_returns = np.log(stocks/stocks.shift(1))
    volatility = log_returns.std(skipna=True)
    rf_rate_df = rf_rates
    if investment_term_days<= 252:
        rate = 0
    else:
        rate = 1
    rf_rate = round(rf_rate_df.iloc[-1,rate]/100,4)

    # Simulating Future Daily Returns
    forecast_period = investment_term_days
    simulations = 5000
    MC_Simulations_df = pd.DataFrame(index=volatility.index)
    MC_Simulations_df.index.name = 'Stock'
    for stock in volatility.index:
        daily_logreturns_simulated = volatility[stock] * norm.ppf(np.random.rand(forecast_period, simulations))
        daily_simplereturns_simulated = np.exp(daily_logreturns_simulated)-1
        # Calculating future price progression in each simulation
        last_price = stocks[stock].iloc[-1]
        price_list = np.zeros_like(daily_simplereturns_simulated)
        price_list[0] = last_price
        MC_Simulations_df.loc[stock,'CurrentPrice'] = round(last_price,2)
        for t in range(1, forecast_period):
            price_list[t] = price_list[t - 1] *(1+ daily_simplereturns_simulated[t])
        #Simulated Price Scenarios
        MC_Simulations_df.loc[stock,'BestPrice'] = round(price_list[-1].max(),2)
        MC_Simulations_df.loc[stock,'AvgPrice'] = round(price_list[-1].mean(),2)
        #MC_Simulations_df.loc[stock,'LikelyPrice'] = statistics.mode(price_list[-1])
        MC_Simulations_df.loc[stock,'LeastPrice'] = round(price_list[-1].min(),2)
        
        #Simulated Confidence Intervals
        num_success = np.sum(price_list[-1] >= price_list[0] * (1 + target_roi))
        probability_of_success = num_success / simulations
        MC_Simulations_df.loc[stock,'%Prob of Return>=SustainableTarget'] = round(probability_of_success*100,2)

        num_loss = np.sum(price_list[-1] < price_list[0]* (1 - rf_rate))
        loss_probability = num_loss / simulations
        MC_Simulations_df.loc[stock,'%Prob of Return>RiskFreeRate'] = round((1-loss_probability)*100,2)

        MC_Simulations_df.to_csv('./Resources/stock_mcs_data.csv',index=True)
#     return(MC_Simulations_df)

# #VaR Calculations for Investments
# def stock_VaR(investment_amount,investment_term_days):
    inv_value = investment_amount
    rf_rate_df = rf_rates
    if investment_term_days<= 252:
        rate = 0
    else:
        rate = 1
    rf_rate = round(rf_rate_df.iloc[-1,rate]/100,4)

    #stocks_data = stock_price_data()

    stocks = stocks_price_data
    log_returns = np.log(stocks/stocks.shift(1))
    rolling_mean = log_returns.rolling(window=investment_term_days).mean()
    returns = np.exp(rolling_mean*investment_term_days)-1
    vol = returns.std(skipna=True)
    
    # log_returns = np.log(stocks/stocks.shift(1))
    # STD = log_returns.std(skipna=True)
    # vol = STD*np.sqrt(investment_term_days)
    
    percentiles = [10,5,1]
    simulations = 5000
    time_period = investment_term_days/252

    VaR_df = pd.DataFrame(index=vol.index)

    for stock in vol.index:
        end_value = inv_value * np.exp((rf_rate - .5 * vol[stock] ** 2) * time_period + np.random.standard_normal(simulations) * vol[stock] * np.sqrt(time_period))
        simulated_inv_returns = end_value - inv_value
        for i in percentiles:
            value = np.percentile(simulated_inv_returns, i)
            VaR_df.loc[stock,f"VaR@{100-i}%"] = value
            #VaR_df.loc[stock,f"Percentage_VaR @{100-i}%"] = round(value*100/investment,2)
    VaR_df.to_csv('./Resources/stock_var_data.csv',index=True)
    #print('PROCEED')
    return(stocks_price_data)