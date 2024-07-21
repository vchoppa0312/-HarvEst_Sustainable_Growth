import streamlit as st
from dataclasses import dataclass
from typing import Any, List
import datetime as datetime
import pandas as pd
import time


def defaults():
    st.session_state.my_values = {'age' : "Less than 35 years",'annualIncome' : 30000,
                                  'investableCashAsset' : 5000, 'targetRoI' : 6.0}
def update_value(key):
    st.session_state.my_values[key] = st.session_state[key]

if 'my_values' not in st.session_state:
    defaults()


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
    expectedRoI: float

# Function to collect client information
def collect_client_info():
    st.title('Client Information')

    name = st.text_input("Name")
    age = st.selectbox(
        "Please Select your age",
        ("Select","Less than 35 years", "35 to 44 years", "45 to 54 years", "55 to 64 years", "65 years and over" ),key='ageBracket')
    country = st.selectbox("Country",("Canada","Other"))
    annualIncome = st.number_input("Annual Income", min_value=0, value=0,step=500, key='income')  
    investableCashAsset = st.slider("Select Amount to Invest", min_value = 5000, max_value=200000, value=0, step=2500,key='invest')
    clientRiskAppetite = st.selectbox(
        "Please Select your Risk Appetite",
        ("Select","High","Medium","Low"))
    investmentTerm = st.selectbox(
                "Please Select your investment term:",
                ("Select","Short(<1 year)", "Medium(1-3 years)", "Long(>3 years)"))
    expectedRoI = st.number_input("Target Percentage Return on Investment",min_value=1.0, value=1.0,step=0.5, key='RoI')
    
    return Client(name,age,country,annualIncome,investableCashAsset,clientRiskAppetite,investmentTerm,expectedRoI)
    
def process_client_info(client):
  
    client_df = pd.DataFrame(vars(client),index=[client.name])
    client_df.to_csv('./Resources/client_info.csv',index=True)

    return client_df

# Main function
def app():
    client_info = collect_client_info()
    left, right = st.columns(2)

    if right.button("Sustainable Growth"):
        client_data = process_client_info(client_info)
        st.switch_page("pages/2_Investment_Info.py")
    if left.button("Home Page"):
        st.switch_page("App_Interface.py")

#         if left.button("Submit"):
# #         st.write("Client Information:")
#         client_data = process_client_info(client)
# #         st.write(client_data)
# #         st.write("Proceed to Client Summary")

if __name__ == "__main__":
    app()