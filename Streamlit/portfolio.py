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
from forex import extract_fx_data

#using selected stock and etf tickers list to get the portfolio's securities price data and log return data
def portfolio_price_read(stock_selected, etf_selected, start_date, end_date):
    stocks_df = yf.download(stock_selected, start = start_date, end = end_date)['Close']
    etf_df = yf.download(etf_selected, start = start_date, end = end_date)['Close']
    portfolio_df = pd.concat([stocks_df, etf_df],axis=1)
    portfolio_returns_df = np.log(portfolio_df).diff()
    portfolio_df.dropna(inplace=True)
    return (portfolio_df,portfolio_returns_df)

#input df is the log return dataframe of the portfolio
def eaqual_weight(df):
    N = len(df.columns)
    weights = np.ones(N)/N
    equal_weighted_returns = df.dot(weights)
    return equal_weighted_returns

def minimum_variance(df):
    N = len(df.columns)
    cov_matrix = df.cov()
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    ones_vestor = np.ones(N)
    weights = inv_cov_matrix.dot(ones_vestor)/ones_vestor.dot(inv_cov_matrix).dot(ones_vestor)
    minimum_var_returns = df.dot(weights)
    return minimum_var_returns

def risk_parity(df):
    N = len(df.columns)
    cov_matrix = df.cov()
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    ones_vestor = np.ones(N)
    weights = inv_cov_matrix.dot(ones_vestor)/N
    risk_parity_returns = df.dot(weights)
    return risk_parity_returns

def cumulative_returns(df):
    cumulative_return = np.exp(df.cumsum())-1
    return cumulative_return

    



    
