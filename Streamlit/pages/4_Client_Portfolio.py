import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as numpy
import yfinance as yf
import streamlit.components.v1 as components
import numpy as np
from scipy.optimize import minimize
from portfolio_builder import stock_portfolio_builder
from portfolio_builder import portfolio_optimization
from portfolio_builder import extract_portfolio_data
from scipy.stats import norm
import plotly.express as px
import os

import warnings
warnings.filterwarnings('ignore')

from pandas.tseries.offsets import DateOffset
from datetime import datetime

st.title('STOCK PORTFOLIO CONSTRUCTION')


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
    st.write('***Portfolio Selection Based on Risk Appetite***')

    form= st.form(key='Portfolio')
    form.write("Stocks within each portfolio ranked according to Treynor Ratio for Risk Optimization")
    portfolio = form.selectbox("**Choose Portfolio Type**",("Select","Conservative","Balanced","High_Risk","Very_High_Risk"))
        #form.divider()
    num_stocks = form.slider("**Select Number of Stocks to be added to the Portfolio**",min_value=5,max_value=20,value=5,step=1)
    form.write("Choose minimum of 5 Stocks for benefits of diversification")
    submit_form = form.form_submit_button(label="***Construct***")

    if submit_form:    
        grid = st.columns(2)
        with grid[0]:
                st.write("Stock Weights allocation based on Return Optimization & Sharpe Ratio Optimization ")
                portfolio_dict = stock_portfolio_builder(investmentTermDays,target_roi)
                stock_dict= dict(list(portfolio_dict[portfolio].items())[:num_stocks])
                portfolio_data = extract_portfolio_data(stock_dict)
                weights_df,results_df = portfolio_optimization(num_stocks,portfolio_data,investmentTermDays,target_roi)
                st.dataframe(results_df)
                
                df = weights_df[['SR_Optimization_Wts']]
                fig = px.pie(df, values='SR_Optimization_Wts', names=df.index,title='Sharpe Ratio Optimized Portfolio Weights')
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
        with grid[1]:
                df2 = weights_df[['Return_Optimization_Wts']]
                fig2 = px.pie(df2, values='Return_Optimization_Wts', names=df.index,title='Return Optimized Portfolio Weights')
                fig2.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig2, use_container_width=True)



    if st.button('Previous Page'):
         st.switch_page("pages/3_Recomendations.py")

if __name__ == "__main__":
    main()
