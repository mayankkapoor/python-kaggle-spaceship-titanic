from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

app = Flask(__name__)

def preprocess_data(data):
    # Preprocessing and feature engineering steps
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Cabin'].fillna('Unknown', inplace=True)
    
    # Fill missing values in CryoSleep with mode
    cryosleep_mode = data['CryoSleep'].mode().iloc[0]
    data['CryoSleep'].fillna(cryosleep_mode, inplace=True)
    data['CryoSleep'] = data['CryoSleep'].astype(int)

    data['TotalSpent'] = data['RoomService'] + data['FoodCourt'] + data['ShoppingMall'] + data['Spa'] + data['VRDeck']
    data['HasCabin'] = data['Cabin'].apply(lambda x: 0 if x == 'Unknown' else 1)
    data['Deck'] = data['Cabin'].apply(lambda x: x[0] if x != 'Unknown' else 'Unknown')

    data.drop(['PassengerId', 'Cabin', 'Name'], axis=1, inplace=True)
    data = pd.get_dummies(data, drop_first=True)

    # Fill missing values in numeric columns with median
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        data[column].fillna(data[column].median(), inplace=True)
    
    # Fill missing values in categorical columns with 'Unknown'
    categorical_columns = data.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        data[column].fillna('Unknown', inplace=True)

    return data

@app.route('/api/predict', methods=['POST'])
def predict():
    input_data = request.get_json()
    data = pd.DataFrame(input_data, index=[0])
    data = preprocess_data(data)
    prediction = model.predict(data)
    return jsonify({"prediction": int(prediction[0])})

if __name__ == '__main__':
    # make sure to copy the model.pkl file into this flask app's folder as model.pkl before running the app
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    app.run(host='0.0.0.0', port=8000)