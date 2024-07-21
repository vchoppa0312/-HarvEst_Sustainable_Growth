from streamlit_option_menu import option_menu
#import Streamlit.images.Investment_Information as Investment_Information, Streamlit.images.Client_Info as Client_Info
import streamlit as st
from dataclasses import dataclass
from typing import Any, List
import datetime as datetime
import pandas as pd
import time
import Investment_Information, Client_Info




st.set_page_config(page_title="HarvEst", page_icon=":tada:", layout="wide")

class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, function):
        self.apps.append({
            "title": title,
            "function": function
        })

    def run(self):
        with st.sidebar:
            app = option_menu(
                menu_title='@HarvEst',
                options=['Investment_Information', 'Client_Info'],
                icons=['person-circle', 'trophy-fill'],
                menu_icon='chat-text-fill',
                default_index=1,
                styles={
                    "container": {"padding": "5!important", "background-color": "green"},
                    "icon": {"color": "white", "font-size": "23px"},
                    "nav-link": {"color": "white", "font-size": "20px", "text-align": "left", "margin": "0px"},
                    "nav-link-selected": {"background-colour": "#02ab21"},
                }
            )

            for a in self.apps:
                if app == a['title']:
                    a['function']()

# Create an instance of MultiApp
app = MultiApp()

# Add your apps using add_app method
app.add_app("Client_Info", Client_Info.py)
app.add_app("Investment_Information", Investment_Information.py)

# Run the apps
app.run()

