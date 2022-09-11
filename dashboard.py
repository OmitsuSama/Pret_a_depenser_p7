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

        tab1, tab2, tab3, tab4 = st.tabs(["Avis", "Infos Clients", "Graphique", "Plus d'info"])
        with tab1:
            if prediction == 0 :
                prob, pred = st.columns(2)
                with prob :
                    st.success(f"Probabilités de remboursement : {round(proba[0][0]*100, 2)} %")
                with pred :
                    st.markdown("<h2 style='text-align: center; color: #44be6e;'>AVIS FAVORABLE</h2>", unsafe_allow_html=True)
            else :
                prob, pred = st.columns(2)
                with prob :
                    st.error(f"Probabilités de remboursement : {round(proba[0][0]*100, 2)} %")
                with pred :
                    st.markdown("<h2 style='text-align: center; color: #ff3d41;'>AVIS DÉFAVORABLE</h2>", unsafe_allow_html=True)
            
            st.markdown("***")

        with tab2:
            age, revenus, montant_credit, montant_annuite = st.columns(4)

            age.metric(label="Age", value=f"{abs(int(round(X.DAYS_BIRTH/365,25)))} ans")
            revenus.metric(label="Revenus annuels", value=f"{int(round(X.AMT_INCOME_TOTAL))} $")
            montant_credit.metric(label="Crédit demandé", value= f"{int(round(X.AMT_CREDIT))} $")
            montant_annuite.metric(label="Montant des annuités", value=f"{int(round(X.AMT_ANNUITY))} $")

            st.markdown("***")

        with tab3:
            # Initialisation des valeurs et des paramètres :
            labels=["Revenu restant","Remboursement crédit"]
            colorlist = ['w','w']
            values=[X["AMT_INCOME_TOTAL"].iloc[0] - X["AMT_ANNUITY"].iloc[0] , X["AMT_ANNUITY"].iloc[0]]
            colors = ["#CBCE91FF", "#EA738DFF"]
            explode = (0,0.2)

            fig = px.pie(values=values, names=["Revenu restant","Remboursement crédit"], title='Impact du crédit sur les revenus', color=["Revenu restant","Remboursement crédit"],
             color_discrete_map={'Revenu restant':'#CBCE91FF',
                                 'Remboursement crédit':'#EA738DFF'})
            st.write(fig)

            all_features, score1 = st.columns(2)

            st.markdown("***")

        with tab4:
            soustab1, soustab2 = st.tabs(["Analyse Bivariée", "Analyse Univariée"])

            with soustab1:    
                def interactive_plot():
                    col1, col2 = st.columns(2)
                    
                    x_axis_val = col1.selectbox('Select the X-axis', options=["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "PAYMANT_RATE", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_BIRTH"])
                    y_axis_val = col2.selectbox('Select the Y-axis', options=["EXT_SOURCE_2", "EXT_SOURCE_1", "EXT_SOURCE_3", "PAYMANT_RATE", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_BIRTH"])

                    plot = px.scatter(data, x=x_axis_val, y=y_axis_val, color='TARGET')
                    st.plotly_chart(plot, use_container_width=True)
                interactive_plot()

            with soustab2:
                x_val = st.selectbox(
                    'Selection du X', ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "PAYMANT_RATE", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_BIRTH"]
                    )            
                x1 = data[data["TARGET"] == 0][x_val]
                x2 = data[data["TARGET"] == 1][x_val]
                x2 = data[data["TARGET"] == 1][x_val]
                group_labels = ['0', '1']

                colors = ['slategray', 'magenta']

                # Create distplot with curve_type set to 'normal'
                fig = ff.create_distplot([x1, x2], group_labels, bin_size=.5,
                                        colors=colors, show_hist=False, show_rug=False)
                x3= data[data['SK_ID_CURR'] == id_client][x_val]
                fig.add_vline(x=x3[0], line_color="green", annotation_text="Individu", annotation_position="top right")
                # fig = px.histogram(data, x=x_val, color=data["TARGET"])
                st.write(fig)



        
        with all_features :
            id_filter_features = st.selectbox("Toutes informations", pd.unique(X.columns))
            feature = X[id_filter_features]
            all_features.metric(label=id_filter_features, value=feature)

        with score1 :
            
            fig = make_subplots(rows=1, cols=3, specs=[[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}]])
            
            labels = ["Non couverts", "Couverts"]
            if X["EXT_SOURCE_1"].isna().values :
                st.markdown("<h3 style='text-align: center; color: #ffffff;'>Score 1 non renseigné</h3>", unsafe_allow_html=True)
            else :
                val = [(1-float(X["EXT_SOURCE_1"])), float(X["EXT_SOURCE_1"])]
                val.append(sum(val))
                colors = ["#CBCE91FF", "#EA738DFF"]
                fig.add_trace(go.Pie(labels=labels, values=val[:-1], hole=.6, marker_colors=colors), 1, 1)

            if X["EXT_SOURCE_2"].isna().values :
                st.markdown("<h3 style='text-align: center; color: #ffffff;'>Score 2 non renseigné</h3>", unsafe_allow_html=True)
            else :
                val = [(1-float(X["EXT_SOURCE_2"])), float(X["EXT_SOURCE_2"])]
                val.append(sum(val))
                colors = ["#CBCE91FF", "#EA738DFF"]
                fig.add_trace(go.Pie(labels=labels, values=val[:-1], hole=.6, marker_colors=colors), 1, 2)
            
            if X["EXT_SOURCE_3"].isna().values :
                st.markdown("<h3 style='text-align: center; color: #ffffff;'>Score 3 non renseigné</h3>", unsafe_allow_html=True)
            else :
                val = [(1-float(X["EXT_SOURCE_3"])), float(X["EXT_SOURCE_3"])]
                val.append(sum(val))
                colors = ["#CBCE91FF", "#EA738DFF"]
                fig.add_trace(go.Pie(labels=labels, values=val[:-1], hole=.6, marker_colors=colors), 1, 3)
            fig.update_layout(annotations=[dict(text='Score 1', x=0.08, y=0.5, font_size=20, showarrow=False),dict(text='Score 2', x=0.5, y=0.5, font_size=20, showarrow=False),dict(text='Score 3', x=0.92, y=0.5, font_size=20, showarrow=False)])
            st.write(fig)

except URLError as e:
    st.error(
        """
        **This demo requires internet access.**
        Connection error: %s
    """
        % e.reason
    )
