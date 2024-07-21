import streamlit as st
from dataclasses import dataclass
from typing import Any, List
import datetime as datetime
import pandas as pd
from bokeh.plotting import figure
import time
import os
st.set_page_config(page_title= "Client Information", page_icon=":tada:", layout="wide")


# Define Client class
@dataclass
class Client:
    name:str
    age: str       
    country: str
    annualIncome: int
    investableCashAsset: int
    clientRiskAppetite: str
    investmentTerm: str


# Function to collect client information
def collect_client_info():
    grid = st.columns(4)
    with grid[0]:
        name = st.text_input("Name", key="name")
        investableCashAsset = st.slider("Select Savings to Invest", min_value = 5000, max_value=200000, value=5000, step=2500,key='invest')
            
    with grid[1]:
        age = st.selectbox(
            "Please Select your age",
            ("Select","Less than 35 years", "35 to 44 years", "45 to 54 years", "55 to 64 years", "65 years and over" ),key='ageBracket')
        clientRiskAppetite = st.selectbox(
            "Please Select your Risk Appetite",
            ("Select","Conservative","Balanced","High","Very High"), key="risk")
        
    with grid[2]:
        country = st.selectbox("Country",("Canada","Other"), key="country")
        investmentTerm = st.selectbox(
                    "Please Select your investment term:",
                    ("Select", "Long (>3 years)", "Medium (1-3 years)","Short (<1 year)"), key="term")
    with grid[3]:
         annualIncome = st.number_input("Annual Income",value=None, key='income',placeholder="Yearly Income")
         
         
    
    return Client(name,age,country,annualIncome,investableCashAsset,clientRiskAppetite,investmentTerm)
    
def process_client_info(client):
  
    client_df = pd.DataFrame(vars(client),index=[client.name])
    client_df.to_csv('./Resources/client_info.csv',index=True)

    return client_df

def economic_data_charts():
    raw_data = pd.read_csv("./Resources/household_economic_data.csv",index_col = 'REF_DATE')
    raw_data = raw_data[['GEO','Characteristics','Income, consumption and savings','VALUE']]
    raw_data = raw_data.loc[raw_data['Income, consumption and savings'].isin(['Household disposable income','Household final consumption expenditure (HFCE)','Household net saving'])]
    raw_data = raw_data.rename_axis('Year')
    raw_data = raw_data.loc[raw_data['GEO']=='Canada']
# Income Quintiles (Lowest income quintile, Second income quintile, Third income quintile, Fourth income quintile & Highest income quintile)
# & Age (Less than 35 years, 35 to 44 years, 45 to 54 years, 55 to 64 years, 65 years and over)
    raw_data = raw_data.loc[raw_data['Characteristics']=='All households']
    raw_data = raw_data.reset_index()
    raw_data = raw_data.pivot(index = 'Year', columns = 'Income, consumption and savings', values = 'VALUE')
    raw_data = raw_data.rename_axis('Financials', axis = 'columns')
    raw_data = raw_data.rename(columns= {'Household disposable income':'Disposable_Income','Household final consumption expenditure (HFCE)':'Consumption_Expenditure','Household net saving':'Net_Savings'}).dropna()

    df_1 = raw_data.loc['2000':'2011'].diff()
    df2_2 = raw_data.loc['2010':'2023'].diff()

    return(df_1,df2_2)


# Main function
def main():
    #st.title("About You")
    #col1, col2 = st.columns(2) #2, gap = "large"
    
    #with col1:
        st.title('Your Information')
        st.write("###### *Fill up the details for interesting insights!*")
        with st.form(key='client'):
            client_info = collect_client_info()
            #client_data = process_client_info(client_info)
            submit_button_r = st.form_submit_button('Submit')
            st.write("Press ***Submit*** before moving to next page")
            if submit_button_r:
                st.write("Scroll Down for Information")
                st.subheader('*Essense of Sustainable Growth*')
                client_data = process_client_info(client_info)
                hv1,hv2 = economic_data_charts()
                st.write('Year Over Year Change in Income, Expenses & Net Savings : 2001-2010')
                st.bar_chart(hv1)
                st.write('Year Over Year Change in Income, Expenses & Net Savings : 2010-2023')
                st.bar_chart(hv2)
                with st.expander("##### *Why Sustainable Growth?*"):

                    explaination = '''
                - From the charts it is evidently visible that for a typical household while ***disposable income and consumption expenditure
                have seen a steady growth year on year***, the **net savings have declined most often** than not, more so in the last decade. 
                - At this rate the only logical explanation funding the rise in cosumption levels is decreased savings or increased debt. ***Factoring inflation*** in the mix, 
                at these unsustainable levels the inevitable is ***depletion of savings*** or ***debt trap***.
                - Now is the best time than ever to understand your economic growth, budget your income and expenses and put your
                investments on a ***Sustainable Growth***.
                - This is where we provide you with a ***Minimum Sustainable Growth Target*** for your investentable assets to aim at, along with a plethora of ***investment choices***
                so as to enjoy sustainable savings and steer wide of debt spirals, all within your choosen risk profile.
                - Insightful Right! To learn more about your growth journey move to ***Next Page*** "''' 
                    st.markdown(explaination)
                    st.divider()
                    st.markdown('''***Hold On!*** As a token of appreciation for being part of this ***Sustainable Growth Journey***, 
                                visit our ***Discovery page*** to get your blockchain wallet and harvest some tokens to spend along this journey with us.''')
            #if submit_button_r:
                    st.divider()
        if st.button('Next Page'):
                        st.switch_page("pages/2_Investment_Info.py")
        st.write("**Thank you !**") 
        if st.button('Discovery'):
                        st.switch_page("pages/6_Discovery.py")



if __name__ == "__main__":
    main()







