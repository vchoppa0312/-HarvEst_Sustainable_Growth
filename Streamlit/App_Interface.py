import streamlit as st
from dataclasses import dataclass
from typing import Any, List
import datetime as datetime
import pandas as pd
import hvplot.pandas
import time
# from streamlit_option_menu import option_menu
# import Investment_Information, Client_Info

st.set_page_config(page_title= "HarvEst", page_icon=":tada:", layout="wide")


# left, right = columns(2)

col1, col2 = st.columns([3,1], gap ="large") #2, gap = "large"



with col1:
    st.title('@HarvEst :copyright:')
    st.header('Investment, Trading & Sustainable Growth: Using AI & ML Models')

#with col2:
    #st.image('image.png')

st.subheader('About Us')
about_us = ''' We, **@HarvEst**, are a group of FinTech enthusiasts from diverse backgrouds coming together
with a goal to leverage latest technological developments in providing informed financial choices
for your investment and trading needs with an inclination to sustainable economic growth.'''

st.markdown(about_us)

st.subheader('What We Do')
what_we_do = '''We use client's information and their expectations on their current financial status to understand
their risk profile, use a variety of AI & ML tools and models to analyse select asset classes and answer
queries on **sustainable growth targets** and **investment choices** from the most important perspective i.e. **$YOU$**'''

st.markdown(what_we_do)

st.subheader('Our Goal')
our_goal = ''' Our goal is to lead you to finanical stability and grow your wealth.'''

# by understanding/learning your income
# and exßpenses w'''

st.markdown(our_goal)

with st.container():
    st.write("---")
    # left_column, right_column = st.columns(2)
    # with left_column:
    #     st.header("picture with a blurb of oursleves?")
    #     st.write("###")

            
st.write("Dive in with your inputs for an insightful journey!")
if st.button('Bon Voyage !'):
    st.switch_page("pages/1_Client_Information.py")
#st.write('__coming soon__')
#if st.button("Obtain Token"):
#    st.switch_page("pages/4_Token.py")
        # latest_iteration = st.empty()
        # bar = st.progress(0)

        # for i in range(100):
        #     latest_iteration.text(f'Iteration {i+1}')
        #     bar.progress(i+1)
        #     time.sleep(0.01)

        # st.write('Congrats we granted you a token')
        # st.balloons()
