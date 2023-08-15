import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from src.util import load_ipo_data, Model

# Streamlit main layout
title = "IPO Analysis"
st.set_page_config(page_title=title, layout="wide", initial_sidebar_state="collapsed")
st.title(title)

ycol_name_mapper = {
    'PctChange1':'First Day Return',
    'Close_Open':'Close-to-Open Return',
    'Open_Ret':'Opening Return',
    '20d_ret':'Next 20 Days Return',
}
# tab_1 = st.tabs(['Model'])

# with tab_1:
cols_a = st.columns(3)
with cols_a[0]:
    # ycol = "PctChange1"
    ycol = st.selectbox(label='Return', options=['PctChange1', 'Close_Open', 'Open_Ret', '20d_ret'], index=0)

with cols_a[1]:
    param_1 = st.number_input('Oversubscription Rate', min_value=0., max_value=1000., value=5.28, step = 0.01, key='or')

with cols_a[2]:
    param_2 = st.number_input('Total Applicants', min_value=0, max_value=100000, value=2603, step = 1, key='ta')

fdf = load_ipo_data()
M = Model(fdf, ycol)
# pred = M.predict(x1=5.28, x2=2603)
pred = M.predict(x1=param_1, x2=param_2)
st.text(f'Prediction (lower bound): {pred[0]:.1%}')
st.text(f'Prediction              : {pred[1]:.1%}')
st.text(f'Prediction (upper bound): {pred[2]:.1%}')

st.plotly_chart(M.fig, use_container_width=True)