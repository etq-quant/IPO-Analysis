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

# with tab_1:
cols_a = st.columns(3)
with cols_a[0]:
    ycol = st.selectbox(label='Return', options=['PctChange1', 'Close_Open', 'Open_Ret', '20d_ret'], index=0)
    include_ace = st.checkbox('Include Ace Market', value=False)
tab_chart, tab_table, tab_pred = st.tabs(['Chart', 'Table', 'Prediction'])

fdf = load_ipo_data(include_ace=include_ace)
stocks = fdf['Stock'].unique()
n = len(stocks)


with tab_chart:
    selected_stocks = st.multiselect(label='Stocks', options=stocks, default=['MSTGOLF', 'CEB'])
    M = Model(fdf, ycol, selected_stocks)
    st.plotly_chart(M.fig, use_container_width=True)

with tab_table:
    st.dataframe(fdf.sort_values('Date', ascending=False), hide_index=True)

with tab_pred:
    cols_b = st.columns(3)
    with cols_b[0]:
        param_1 = st.number_input('Oversubscription Rate', min_value=0., max_value=1000., value=5.28, step = 0.01, key='or')
    with cols_b[1]:
        param_2 = st.number_input('Total Applicants', min_value=0, max_value=100000, value=2603, step = 1, key='ta')
    pred = M.predict(x1=param_1, x2=param_2)
    st.text(f'Prediction (lower bound): {pred[0]:.1%}')
    st.text(f'Prediction              : {pred[1]:.1%}')
    st.text(f'Prediction (upper bound): {pred[2]:.1%}')