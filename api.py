import joblib
from flask import Flask, jsonify
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

data = pd.read_csv('app_test.csv')
print('la taille de Dataframe est = ', data.shape)