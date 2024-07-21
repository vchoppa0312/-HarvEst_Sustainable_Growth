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

def snp500_tickers():
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
    return(stock_ticker_dict)
# Only For Master File Generation
def stock_price_data():
    ticker_dict = snp500_tickers()
    stocks_data = pd.DataFrame()
    for stock,ticker in ticker_dict.items():
        stock_df = yf.download(ticker,start_date,end_date)
        stock_df.index = pd.to_datetime(stock_df.index)
        stock_df = stock_df[['Close']]
        stock_df = stock_df.rename(columns={'Close':stock})
        stocks_price_data = pd.concat([stocks_data, stock_df],axis=1)
    stocks_price.to_csv('./Resources/stocks_price_data.csv',index=True)
    return(stocks_price_data)

# # Extract Stock Price for Select Stock:Ticker Pair
# def extract_stock_data(ticker_dict):
#     # Input dictionary with {stock_name:ticker_symbol}
#     stocks_data = pd.DataFrame()
#     for stock,ticker in ticker_dict.items():
#         stock_df = yf.download(ticker,start_date,end_date)
#         stock_df.index = pd.to_datetime(stock_df.index)
#         stock_df = stock_df[['Close']]
#         stock_df = stock_df.rename(columns={'Close':stock})
#         stocks_data = pd.concat([stocks_data, stock_df],axis=1)

#     return(stocks_data)

# Extract Stock Fundamentals Select Stock:Ticker Pair
def stock_key_financials():
    ticker_dict = snp500_tickers()
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
        stock_fundamentals.loc[stock,'SMA50'] = key_info['fiftyDayAverage']
        stock_fundamentals.loc[stock,'SMA200'] = key_info['twoHundredDayAverage']
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
    return(stock_fundamentals)

# Extracting Risk Free Rates
def risk_free_rates():
    rf_dict = {'13W_Risk_Free':'^IRX','10Y_Risk_Free':'^TNX'}
    rf_rates = pd.DataFrame()
    for key, value in rf_dict.items():
        rf_rates_df = yf.download(value,start_date,end_date)
        rf_rates_df.index = pd.to_datetime(rf_rates_df.index)
        rf_rates[key] = rf_rates_df['Close']

    return(rf_rates)

# Sharpe Ratio
def stock_sharpe_ratio(investment_term_days):
    stocks_data = pd.read_csv('./Resources/stocks_price_data.csv',index_col='Date',
                              parse_dates=True,infer_datetime_format=True)
    #stocks_data = stock_price_data()
    rf_rate = risk_free_rates()
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

    return(Sharpe_Ratio)

#MCSimulation of Stock Returns
def stock_mc_simulation(investment_term_days,target_roi):
    stocks = pd.read_csv('./Resources/stocks_price_data.csv',index_col='Date',
                         parse_dates=True,infer_datetime_format=True)
    #stocks_data = stock_price_data()
    log_returns = np.log(stocks/stocks.shift(1))
    volatility = log_returns.std(skipna=True)
    rf_rate_df = risk_free_rates()
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
    return(MC_Simulations_df)

#VaR Calculations for Investments
def stock_VaR(investment_amount,investment_term_days):
    inv_value = investment_amount
    rf_rate_df = risk_free_rates()
    if investment_term_days<= 252:
        rate = 0
    else:
        rate = 1
    rf_rate = round(rf_rate_df.iloc[-1,rate]/100,4)

    #stocks_data = stock_price_data()

    stocks = pd.read_csv('./Resources/stocks_price_data.csv',index_col='Date',
                               parse_dates=True,infer_datetime_format=True)

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
    
    return(VaR_df)

def stock_classification():
    stock_fundamentals_df = pd.read_csv('./Resources/stock_fundamentals_data.csv',index_col='Stock')
    #stock_fundamentals_df = stock_key_financials()
    sr_df = pd.read_csv('./Resources/stock_sharpe_ratio.csv',index_col='Stock')
    mcs_df = pd.read_csv('./Resources/stock_mcs_data.csv',index_col='Stock')
    var_df = pd.read_csv('./Resources/stock_var_data.csv',index_col='Stock')
    
    stock_classifier = pd.concat([stock_fundamentals_df,sr_df,mcs_df,var_df],axis=1)
    stock_classifier.index.name='Stock'

    #Estimated Returns over investment term using MC_Simulation
    stock_classifier['MeanExpectedReturn'] = (stock_classifier['AvgPrice']-stock_classifier['CurrentPrice'])/stock_classifier['CurrentPrice']
    stock_classifier['Expected_Annual_Return(MCS)'] = ((1+stock_classifier['MeanExpectedReturn'])**(252/investment_term_days))-1
    #Treynor Ratio
    #stock_classifier.loc['S&P 500','Beta']=1.00
    stock_classifier['Treynor_Ratio'] = (stock_classifier['Sharpe_Ratio']*stock_classifier['avg_volatility'])/stock_classifier['Beta']

    stock_classifier["CompanySize"] = pd.cut(stock_classifier["MarketCap(BN)"],bins=[0, 0.25, 2.0, 10.0, 200.00,5000.00], labels=["Micro", "Small", "Mid", "Large","Mega"])
    stock_preference = ['Least Preferred','Neutral','Preferred','Most Preferred']
    stock_classifier['Return_Profile'] = pd.qcut(stock_classifier['Avg_Annual_Returns'],4,stock_preference)
    stock_classifier['TreynorRatio_Profile']=pd.qcut(stock_classifier['Treynor_Ratio'],4,stock_preference)
    stock_classifier['SharpeRatio_Profile'] = pd.qcut(stock_classifier.loc[stock_classifier["Sharpe_Ratio"]>0,'Sharpe_Ratio'],4,stock_preference)
    stock_classifier['Risk_Profile'] = pd.qcut(stock_classifier['Avg_Annual_Volatility'],4,['Most Preferred','Preferred','Neutral','Least Preferred'])
    stock_classifier['ProfitMargin_Profile']=pd.cut(stock_classifier['ProfitMargins'],bins=[-5,0.0,0.1,0.2,5],labels=['Avoid','Low','Healthy','High'])

    #stock_classifier.to_csv('./Resources/stock_classification_data.csv',index=True)
    return(stock_classifier)

def stock_risk_profiling():
    
    stock_classifier = stock_classification(investment_amount,investment_term_days,target_roi)
    stock_data_df = stock_classifier[['Ticker','Sector','Industry','CompanySize','CurrentPrice','SMA50','SMA200']]
    stock_analysis_df = stock_classifier[['Beta','Avg_Annual_Volatility','Avg_Annual_Returns','ProfitMargins','Treynor_Ratio','Sharpe_Ratio']]
    stock_profile_df = stock_classifier[['Risk_Profile','Return_Profile','ProfitMargin_Profile','TreynorRatio_Profile','SharpeRatio_Profile']]
    stock_predictions_df = stock_classifier[['AvgPrice','%Prob of Return>=SustainableTarget',
                                         '%Prob of Return>RiskFreeRate','VaR@99%']]
    return(stock_data_df,stock_analysis_df,stock_profile_df,stock_predictions_df)

def stock_portfolio_buider(investment_amount,investment_term_days,target_roi):
    stock_classifier = stock_classification(investment_amount,investment_term_days,target_roi)
    select_stocks = stock_classifier.loc[(stock_classifier['Avg_Annual_Returns']>=target_roi/100)]
    risk_profile_dict = {'VeryHighRisk':'Least Preferred','HighRisk':'Neutral',
                     'Balanced':'Preferred','Conservative':'Most Preferred'}
    stock_lists_dict = {}
    for key,item in risk_profile_dict.items():
        # Filter 1 - Risk_Profile
        filter1_df = select_stocks.loc[(select_stocks['Risk_Profile']==item)]
        # Filter 2 - Treynor_Profile
        filter2_df = filter1_df.loc[(filter1_df['TreynorRatio_Profile']=='Most Preferred')|(filter1_df['TreynorRatio_Profile']=='Preferred')]
        # Filter 3 - ProfitMargin_Profile & SR_Profile
        filter3_df = filter2_df.loc[((filter2_df['ProfitMargin_Profile']=='High')|(filter2_df['ProfitMargin_Profile']=='Healthy'))&((filter2_df['SharpeRatio_Profile']=='Most Preferred'))]
        # Sort by Annual Returns
        stocks = filter3_df.sort_values('Avg_Annual_Returns',ascending=False)

        stock_lists_dict[f"{key}"] = stocks.index.to_list()

    return(stock_lists_dict)