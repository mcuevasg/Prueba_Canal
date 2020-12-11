# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 18:48:58 2020

@author: mcuev
"""

import pandas as pd
import streamlit as st

lqh=pd.read_csv('LQH_ST.csv')

st.write("""
# Base Lugares que Hablan
Muestra la base del programa LQH cons sus características
""")

st.write(lqh)