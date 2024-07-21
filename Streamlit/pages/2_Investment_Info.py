
#Import Libaries
import streamlit as st
from dataclasses import dataclass
from typing import Any, List
import numpy as np
import pandas as pd
import hvplot.pandas
import os

from sklearn import linear_model
from stockMasterData import snp500_master_data
import warnings
warnings.filterwarnings('ignore')


st.set_page_config(page_title= "Investment Information", page_icon=":tada:", layout="wide")

def defaults():
    st.session_state.my_values = {'age' : "Less than 35 years",'annualIncome' : 30000,
                                  'investableCashAsset' : 5000}
def update_value(key):
    st.session_state.my_values[key] = st.session_state[key]

if 'my_values' not in st.session_state:
    defaults()

client_info = pd.read_csv("./Resources/client_info.csv",index_col='name')
annual_income = client_info['annualIncome'][0]
investment_amount = client_info['investableCashAsset'][0]
investmentTerm = client_info['investmentTerm'][0]
age_bracket = client_info['age'][0]
clientRiskAppetite= client_info['clientRiskAppetite'][0]
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

@st.cache_data
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
    RoI={}
    regr = linear_model.LinearRegression()
    X = income_based.drop(columns=['Sustainable_Inc_Growth_Rate_Target'])
    y = income_based['Sustainable_Inc_Growth_Rate_Target']
    regr.fit(X,y)
    #coeff = regr.coef_
    #intercept = regr.intercept_
    inc_growth_target = regr.predict(np.array([inc, exp ,inf]).reshape(1,-1))[0]
    amount_inc = annual_income*inc_growth_target/100
    roi_target_inc = round((amount_inc/investment_amount)*100,2)
    #inv_amount_inc = round((amount_inc*100/expected_roi),0)

    X = age_based.drop(columns=['Sustainable_Inc_Growth_Rate_Target'])
    y = age_based['Sustainable_Inc_Growth_Rate_Target']
    regr.fit(X,y)
    #coeff = regr.coef_
    #intercept = regr.intercept_
    inc_growth_target = regr.predict(np.array([inc, exp ,inf]).reshape(1,-1))[0]
    amount_inc = annual_income*inc_growth_target/100
    roi_target_age = round((amount_inc/investment_amount)*100,2)
    #inv_amount_age = round((amount_inc*100/expected_roi),2)
    RoI['target_roi'] = [round((roi_target_inc + roi_target_age)/2,2)]
    RoI_df = pd.DataFrame.from_dict(RoI)
    RoI_df.to_csv('./Resources/roi.csv',index=True)
    return(roi_target_inc,roi_target_age)


def stock_data(investment_amount,investmentTermDays,target_roi):
    df = snp500_master_data(investment_amount,investmentTermDays,target_roi)
    return(df)

# Main function
def main():   
    st.header('Minimum Sustainable Return on Investment(RoI)') 

    st.subheader('Tell us about your expectations')
    what_we_do = '''This helps us asses your sustainable economic growth targets. We take into
    account your anticipated changes in disposable income and expenditure **(growth or decline in percentage
     terms related to your current annual numbers)** along with latest or projected inflation rates.
    '''
    st.markdown(what_we_do)

    col1,col2 = st.columns([1.75,2],gap="large")
    
    with col1:
        form = st.form(key='client_form')
        form.write(f"*Your Annual Disposable Income*: **${annual_income}**")
        #form.write(f"*Expected Return of Investments*: **{expected_roi}**%")
        form.write(f"*Investable Cash Assets*: **${investment_amount}**") 
        inc = form.number_input("Expected Percentage Change in Annual Income(Year Over Year)",
                                 value=None, placeholder='%Change in Yearly Income',step=0.5)
        #form.caption('What % you would expect your Yearly Income to change by')
        exp = form.number_input("Expected Percentage Change in Consumption Expenditure(Year Over Year)",
                                value=None, placeholder='%Change in Yearly Consumption',step=0.5)
        #form.caption('What % you would expect your Yearly Consumption to change by')
        inf = form.number_input("Projected/Current Inflation",value=None,placeholder= "*Current Inflation Rate 3.40*")
        submit_button_right = form.form_submit_button(label='My Sustainable Growth Targets')

    with col2:
        if submit_button_right:
            st.subheader("Minimum Sustainable Investment Recommendation ")
            income_based_roi,age_based_roi = sustainable_growth_targets(inc,exp,inf)
            #st.info(f"Minimum Sustainable Investment for your income-based risk profile is ${income_based_inv}, for your {expected_roi}% expected Return of Investment.")
            st.divider()
            st.success(f"Minimum Sustainable Annual Return on Investments to target based on your income-profile for a minimun investment of **${investment_amount}** is **{income_based_roi}%**.")
            # st.write(f"Based on your selections you need to hit a min of {income_based_roi} to reach sustainable growth")
            st.divider()
           # st.warning(f"Minimum Sustainable Investment required at {expected_roi}% expected RoI for your age-based risk profile is ${age_based_inv}")
           # st.divider()
            st.info(f"Minimum Sustainable Annual Return on Investments to target based on your age-profile for a minimun investment of **${investment_amount}** is **{age_based_roi}%**.")
            st.divider()
    st.write('*This information is based on 20 Years of Historical Data on Household Disposable Income and Consumption Expenditure*')
    income_based_roi,age_based_roi = sustainable_growth_targets(inc,exp,inf)
    target_roi = (income_based_roi+age_based_roi)/2/100


    if st.button('Continue The Journey'):
         price_data = stock_data(investment_amount,investmentTermDays,target_roi)
         st.switch_page("pages/3_Recommendations.py")
    if st.button('Back'):
         st.switch_page("pages/1_Client_Information.py")
       

        
    
      

if __name__ == "__main__":
    main()

