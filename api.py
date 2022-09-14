import joblib
from flask import Flask, jsonify
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
import warnings
import json
from json import JSONEncoder
import eli5
warnings.filterwarnings("ignore", category=UserWarning)

data = pd.read_csv('app_test.csv')
print('la taille de Dataframe est = ', data.shape)

model = joblib.load(open('Model.joblib', 'rb'))

notimportant_features = ['SK_ID_CURR', 'INDEX', 'TARGET']
selected_features = [col for col in data.columns if col not in notimportant_features]
dict_table = {
        "feature_importance": eli5.explain_weights(model.named_steps['regressor'], top=50, feature_names=selected_features)
}
# print(dict_table)
print(model.__dict__)

app = Flask(__name__)
@app.route("/features")
def features():
    model = joblib.load(open('Model.joblib', 'rb'))
    X = data
    notimportant_features = ['SK_ID_CURR', 'INDEX', 'TARGET']
    selected_features = [col for col in data.columns if col not in notimportant_features]
    
    X = X[selected_features]
    feature_imp = pd.DataFrame(sorted(zip(model.steps[1][1].feature_importances_,X.columns)), columns=['Value','Feature'])
    result = feature_imp.to_json(orient="split")
    parsed = json.loads(result)
    return json.dumps(parsed, indent=4)
@app.route('/')
def hello():
    return "Bienvenue, L'API est opérationnelle... tapez /prediction_credit/{id_dun_client}"

@app.route('/prediction_credit/<id_client>')  #, methods=['GET'])
def prediction_credit(id_client):
    print('id client = ', id_client)
    
    ID = int(id_client)
    X = data[data['SK_ID_CURR'] == ID]
    
    print("Laisse moi tranquille : ", X)

    notimportant_features = ['SK_ID_CURR', 'INDEX', 'TARGET']
    selected_features = [col for col in data.columns if col not in notimportant_features]
    
    X = X[selected_features]
    
    print('La taille du vecteur X  = ', X.shape)
    
    proba = model.predict_proba(X)
    prediction = model.predict(X)
 
    print('L\'identificateur du client : ', id_client)
  
    dict_final = {
        'prediction' : int(prediction),
        'proba' : float(proba[0][0])
        }
   
    print('Lancer une nouvelle Prédiction : \n', dict_final)
            
    return jsonify(dict_final)

          

if __name__ == "__main__":
    app.debug = True
    app.run()