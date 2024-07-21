import streamlit as st
from dataclasses import dataclass
from typing import Any, List
import numpy as np
import pandas as pd
import hvplot.pandas
import os

from sklearn import linear_model
import warnings
warnings.filterwarnings('ignore')




client_info = pd.read_csv("./Resources/client_info.csv",index_col='name')
annual_income = client_info['annualIncome'][0]
investment_amount = client_info['investableCashAsset'][0]
expected_roi = client_info['expectedRoI'][0]
age_bracket = client_info['age'][0]

if annual_income <= 36000:
        income_level = 'Lowest income quintile'
elif annual_income >36000 and annual_income <= 60000:
        income_level = 'Second income quintile'
elif annual_income > 60000 and annual_income <= 88000:
        income_level = 'Third income quintile'
elif annual_income >88000 and annual_income <= 134000:
        income_level = 'Fourth income quintile'
else:
        income_level = 'Highest income quintile'


# Analysis based on income category
def base_financials_income():
    raw_data = pd.read_csv("./Resources/household_economic_data.csv",index_col = 'REF_DATE')
    raw_data = raw_data[['GEO','Characteristics','Income, consumption and savings','VALUE']]
    raw_data = raw_data.loc[raw_data['Income, consumption and savings'].isin(['Household disposable income','Household final consumption expenditure (HFCE)'])]
    raw_data = raw_data.rename_axis('Year')
    raw_data = raw_data.loc[raw_data['GEO']=='Canada']
# Income Quintiles (Lowest income quintile, Second income quintile, Third income quintile, Fourth income quintile & Highest income quintile)
# & Age (Less than 35 years, 35 to 44 years, 45 to 54 years, 55 to 64 years, 65 years and over)
    raw_data = raw_data.loc[raw_data['Characteristics'].isin([income_level])]
    raw_data = raw_data.reset_index()
    raw_data = raw_data.pivot(index = 'Year', columns = 'Income, consumption and savings', values = 'VALUE')
    raw_data = raw_data.rename_axis('Financials', axis = 'columns')
    raw_data = raw_data.rename(columns= {'Household disposable income':'Disposable_Income','Household final consumption expenditure (HFCE)':'Consumption_Expenditure'}).dropna()

    yoy_pct_change = round(raw_data.pct_change()*100,2)
    yoy_pct_change.index = yoy_pct_change.index.astype('int')
    yoy_pct_change = yoy_pct_change.rename(columns={'Disposable_Income':'Disposable_Income_%Change','Consumption_Expenditure':'Consumption_Expenditure_%Change'}).dropna()
#Inflation Data
    inflation_raw_data = pd.read_csv("./Resources/inflation_data.csv")
    inflation_raw_data = inflation_raw_data[['REF_DATE','Alternative measures','VALUE']]
    inflation_raw_data = inflation_raw_data.loc[inflation_raw_data['Alternative measures'].isin(['Measure of core inflation based on a factor model, CPI-common (year-over-year percent change)','Measure of core inflation based on a weighted median approach, CPI-median (year-over-year percent change)','Measure of core inflation based on a trimmed mean approach, CPI-trim (year-over-year percent change)'])].dropna()    
    inflation_raw_data = inflation_raw_data.pivot(index='REF_DATE',columns='Alternative measures',values='VALUE')
    inflation_raw_data = inflation_raw_data.reset_index()
    inflation_raw_data['Year'] = [year.split('-')[0] for year in inflation_raw_data["REF_DATE"]]
    inflation_raw_data = inflation_raw_data.drop(columns=['REF_DATE'])
    inflation_raw_data = inflation_raw_data.rename_axis('Inflation',axis='columns')
    inflation_raw_data = inflation_raw_data.rename(columns = {'Measure of core inflation based on a factor model, CPI-common (year-over-year percent change)':'CPI-common','Measure of core inflation based on a weighted median approach, CPI-median (year-over-year percent change)':'CPI-median','Measure of core inflation based on a trimmed mean approach, CPI-trim (year-over-year percent change)':'CPI-trim'})
    inflation_raw_data = inflation_raw_data.groupby(['Year']).max()
    inflation_raw_data['Inflation'] = inflation_raw_data.max(axis=1)
    inflation_raw_data.index = inflation_raw_data.index.astype('int')

    base_financials = pd.concat([yoy_pct_change,inflation_raw_data[['Inflation']]],join='inner',axis=1).dropna()
#Sustainable Income Growth Rate
    base_financials['Sustainable_Inc_Growth_Rate_Target'] = base_financials['Inflation']+base_financials['Consumption_Expenditure_%Change']-base_financials['Disposable_Income_%Change']
    base_financials['Sustainable_Inc_Growth_Rate_Target']= np.where(base_financials['Sustainable_Inc_Growth_Rate_Target']>0,base_financials['Sustainable_Inc_Growth_Rate_Target'],0)

    return(base_financials)

# Analysis based on the age category
def base_financials_age():
    raw_data = pd.read_csv("./Resources/household_economic_data.csv",index_col = 'REF_DATE')
    raw_data = raw_data[['GEO','Characteristics','Income, consumption and savings','VALUE']]
    raw_data = raw_data.loc[raw_data['Income, consumption and savings'].isin(['Household disposable income','Household final consumption expenditure (HFCE)'])]
    raw_data = raw_data.rename_axis('Year')
    raw_data = raw_data.loc[raw_data['GEO']=='Canada']
# Income Quintiles (Lowest income quintile, Second income quintile, Third income quintile, Fourth income quintile & Highest income quintile)
# & Age (Less than 35 years, 35 to 44 years, 45 to 54 years, 55 to 64 years, 65 years and over)
    raw_data = raw_data.loc[raw_data['Characteristics'].isin([age_bracket])]
    raw_data = raw_data.reset_index()
    raw_data = raw_data.pivot(index = 'Year', columns = 'Income, consumption and savings', values = 'VALUE')
    raw_data = raw_data.rename_axis('Financials', axis = 'columns')
    raw_data = raw_data.rename(columns= {'Household disposable income':'Disposable_Income','Household final consumption expenditure (HFCE)':'Consumption_Expenditure'}).dropna()

    yoy_pct_change = round(raw_data.pct_change()*100,2)
    yoy_pct_change.index = yoy_pct_change.index.astype('int')
    yoy_pct_change = yoy_pct_change.rename(columns={'Disposable_Income':'Disposable_Income_%Change','Consumption_Expenditure':'Consumption_Expenditure_%Change'}).dropna()
#Inflation Data
    inflation_raw_data = pd.read_csv("./Resources/inflation_data.csv")
    inflation_raw_data = inflation_raw_data[['REF_DATE','Alternative measures','VALUE']]
    inflation_raw_data = inflation_raw_data.loc[inflation_raw_data['Alternative measures'].isin(['Measure of core inflation based on a factor model, CPI-common (year-over-year percent change)','Measure of core inflation based on a weighted median approach, CPI-median (year-over-year percent change)','Measure of core inflation based on a trimmed mean approach, CPI-trim (year-over-year percent change)'])].dropna()    
    inflation_raw_data = inflation_raw_data.pivot(index='REF_DATE',columns='Alternative measures',values='VALUE')
    inflation_raw_data = inflation_raw_data.reset_index()
    inflation_raw_data['Year'] = [year.split('-')[0] for year in inflation_raw_data["REF_DATE"]]
    inflation_raw_data = inflation_raw_data.drop(columns=['REF_DATE'])
    inflation_raw_data = inflation_raw_data.rename_axis('Inflation',axis='columns')
    inflation_raw_data = inflation_raw_data.rename(columns = {'Measure of core inflation based on a factor model, CPI-common (year-over-year percent change)':'CPI-common','Measure of core inflation based on a weighted median approach, CPI-median (year-over-year percent change)':'CPI-median','Measure of core inflation based on a trimmed mean approach, CPI-trim (year-over-year percent change)':'CPI-trim'})
    inflation_raw_data = inflation_raw_data.groupby(['Year']).max()
    inflation_raw_data['Inflation'] = inflation_raw_data.max(axis=1)
    inflation_raw_data.index = inflation_raw_data.index.astype('int')

    base_financials = pd.concat([yoy_pct_change,inflation_raw_data[['Inflation']]],join='inner',axis=1).dropna()
#Sustainable Income Growth Rate
    base_financials['Sustainable_Inc_Growth_Rate_Target'] = base_financials['Inflation']+base_financials['Consumption_Expenditure_%Change']-base_financials['Disposable_Income_%Change']
    base_financials['Sustainable_Inc_Growth_Rate_Target']= np.where(base_financials['Sustainable_Inc_Growth_Rate_Target']>0,base_financials['Sustainable_Inc_Growth_Rate_Target'],0)

    return(base_financials)

# Modeling Linear Regression fit for Target Income Investment Growth Rate 
# based on Inflation and Historical Household Income & Consumption trends  
def sustainable_growth_targets(inc,exp,inf):
    income_based = base_financials_income()
    age_based = base_financials_age()
    regr = linear_model.LinearRegression()
    X = income_based.drop(columns=['Sustainable_Inc_Growth_Rate_Target'])
    y = income_based['Sustainable_Inc_Growth_Rate_Target']
    regr.fit(X,y)
    #coeff = regr.coef_
    #intercept = regr.intercept_
    inc_growth_target = regr.predict(np.array([inc, exp ,inf]).reshape(1,-1))[0]
    amount_inc = annual_income*inc_growth_target/100
    roi_target_inc = round((amount_inc/investment_amount)*100,2)
    inv_amount_inc = round((amount_inc*100/expected_roi),0)

    X = age_based.drop(columns=['Sustainable_Inc_Growth_Rate_Target'])
    y = age_based['Sustainable_Inc_Growth_Rate_Target']
    regr.fit(X,y)
    #coeff = regr.coef_
    #intercept = regr.intercept_
    inc_growth_target = regr.predict(np.array([inc, exp ,inf]).reshape(1,-1))[0]
    amount_inc = annual_income*inc_growth_target/100
    roi_target_age = round((amount_inc/investment_amount)*100,2)
    inv_amount_age = round((amount_inc*100/expected_roi),2)

    return(inv_amount_inc,roi_target_inc,inv_amount_age,roi_target_age)

# Main function
def app():    
    st.title('Sustainable Financials: What, Why & How') 

    st.subheader('Tell us about your expectations')



    form = st.form(key='client_form')
    form.write(f"Your Annual Disposable Income: {annual_income}")
    form.write(f"Expected RoI: {expected_roi}%")
    form.write(f"Investable Cash Assets: ${investment_amount}")
    inc = form.number_input("Expected Percentage Change in Annual Income(YoY)",step=0.1)
    exp = form.number_input("Expected Percentage Change in Consumption Expenditure(YoY)",step=0.1)
    inf = form.number_input("Current / Anticipated Inflation",step=0.1)
    left, right = form.columns(2)
    submit_button_right = form.form_submit_button(label='My Sustainable Growth Targets')
    submit_button_left = form.form_submit_button(label='Back')
    
    if submit_button_right:
        st.subheader("Based on Historical Data of Year-on-Year change in Household Disposable Income and Consumption Expenditure:")
        income_based_inv,income_based_roi,age_based_inv,age_based_roi = sustainable_growth_targets(inc,exp,inf)

        st.write(f"Minimum Sustainable Investment required at {expected_roi}% expected RoI for your income-based risk profile is ${income_based_inv}")
        st.divider()
        st.write(f"Minimum Sustainable RoI required on an investment of ${investment_amount} for your income-based risk profile is {income_based_roi}%")
        st.divider()
        st.write(f"Minimum Sustainable Investment required at {expected_roi}% expected RoI for your age-based risk profile is ${age_based_inv}")
        st.divider()
        st.write(f"Minimum Sustainable RoI required on an investment of ${investment_amount} for your age-based risk profile is {age_based_roi}%")
    
        st.snow()

    if submit_button_left:
        st.switch_page("pages/1_Client_Information.py")

      

if __name__ == "__main__":
    app()
