import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from urllib.error import URLError
import altair as alt
import plotly.express as px  
import plotly as py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


st.title('Prêt à dépenser : Dashboard')
@st.cache
def get_UN_data():
    df = pd.read_csv('app_test.csv')
    return df.set_index("SK_ID_CURR")

try:
    df = get_UN_data()
    id_client = st.selectbox(
        "Choisissez un id client :", list(df.index)
    )
    if not id_client:
        st.error("Choisissez un id client.")
    else:
        model = joblib.load(open('Model.joblib', 'rb'))
        data = pd.read_csv('app_test.csv')

        X = data[data['SK_ID_CURR'] == id_client]
        
        notimportant_features = ['SK_ID_CURR', 'INDEX', 'TARGET']
        selected_features = [col for col in data.columns if col not in notimportant_features]
        
        X = X[selected_features]
               
        proba = model.predict_proba(X)
        prediction = model.predict(X)
except URLError as e:
    st.error(
        """
        **This demo requires internet access.**
        Connection error: %s
    """
        % e.reason
    )
