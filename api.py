import joblib
from flask import Flask, jsonify
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

data = pd.read_csv('app_test.csv')
print('la taille de Dataframe est = ', data.shape)

model = joblib.load(open('Model.joblib', 'rb'))

app = Flask(__name__)

@app.route('/')
def hello():
    return "Bienvenue, L'API est opérationnelle..."