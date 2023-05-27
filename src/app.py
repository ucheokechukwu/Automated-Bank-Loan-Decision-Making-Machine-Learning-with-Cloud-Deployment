# import Flask and jsonify

from flask import Flask, jsonify, request
import flask
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

       
        


# include user_defined class and functions
from sklearn.preprocessing import FunctionTransformer
def log_transformer(X):
    X_log = np.log(X)
    return X_log
log_transform = FunctionTransformer(log_transformer)

# create a class 
class IncomeGenerator():
    '''sums Applicantincome and Coapplicantincome and 
        returns as new column Income''' 
    def __init__(self):
        pass
        
    def fit(self, X=None, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        X['Income'] = X['ApplicantIncome'] + X['CoapplicantIncome']
        return X

df_for_customimputer = pickle.load(open('df_for_customimputer', 'rb'))        
class CustomImputer():
    """
    Custom class to fill in the missing values of Married, Dependents, Credit_History and Self_Employed
    Fills in the first 3 with default values
    Fills the last feature with values based on income brackets
    """
    def __init__(self, df=df_for_customimputer):
        # dictionary of missing values
        self.defaults = {'Married': 'No',
              'Dependents': '0', 
              'Credit_History': 0}
        
        df = df[['Self_Employed', 'ApplicantIncome']]
        self.self_employed_median_income = df.groupby('Self_Employed')['ApplicantIncome'].median()   
        self.ApplicantIncome = df['ApplicantIncome']
        
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X = X.copy()
        for col in X.columns:
            if col in self.defaults.keys():
                X[col] = X[col].fillna(self.defaults[col])
            elif col == 'Self_Employed':

                # special case for 'Self_Employed' column  
                
                filter = X[col].isna()
                for idx in X[col][filter].index:
                    # find the nearest average
                    difference = self.self_employed_median_income - self.ApplicantIncome[idx]
                    # Yes if it's nearest to index 1 or No if it's not
                    X.loc[idx, col] = 'Yes' if difference.argmin() else 'No' 
        return X
 
 
# import model and columns list    
model = pickle.load(open('model.pkl', 'rb'))
model_columns = pickle.load (open('columns_list', 'rb'))

 


@app.route('/')
def welcome():
    return "Welcome! Use this Flask App for Bank Loan Prediction"
     
        
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    
    if flask.request.method == 'GET':
        return "Prediction page. Use post with params to get a specific prediction."
        
    if flask.request.method == 'POST':
        try:
            json_ = request.json # 
            print(json_)
            query_ = pd.DataFrame(json_)
            query = query_.reindex(columns = model_columns, fill_value= 0)
            prediction = list(model.predict(query))
            return jsonify({
                "prediction":str(prediction)
            })
        except:
            return jsonify({
                "trace": traceback.format_exc()
            })
            
            
@app.route('/probability', methods=['POST', 'GET'])
def probability():
    
    if flask.request.method == 'GET':
        return "Prediction Probability page. Use post with params to get the specific prediction probabilities."
        
    if flask.request.method == 'POST':
        try:
            json_ = request.json # 
            print(json_)
            query_ = pd.DataFrame(json_)
            query = query_.reindex(columns = model_columns, fill_value= 0)
            N_probability, Y_probability = list(model.predict_proba(query))[0]
            return jsonify({
                "N probability":str(N_probability),
                "Y probability":str(Y_probability)
            })
        except:
            return jsonify({
                "trace": traceback.format_exc()
            })


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8000)