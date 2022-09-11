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
