import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import os

app = Flask(__name__)

# Load Dataset
df = pd.read_csv('Cropprice.csv')
df.ffill(inplace=True)  # Handle missing values

# Load Pretrained Models & Scalers
with open('minmaxscaler.pkl', 'rb') as minmax_file:
    mx = pickle.load(minmax_file)

with open('standscaler.pkl', 'rb') as stand_file:
    sc = pickle.load(stand_file)

with open('model.pkl', 'rb') as model_file:
    randclf = pickle.load(model_file)

xgb_model = pickle.load(open('cropPricePredictionModel.pkl', 'rb'))
nn_model = load_model('nn_model.keras')  # Load neural network model

# Crop Dictionary
crop_dict = {
    1: 'rice', 2: 'maize', 3: 'jute', 4: 'cotton', 5: 'coconut',
    6: 'papaya', 7: 'orange', 8: 'apple', 9: 'muskmelon', 10: 'watermelon',
    11: 'grapes', 12: 'mango', 13: 'banana', 14: 'pomegranate', 15: 'lentil',
    16: 'blackgram', 17: 'mungbean', 18: 'mothbeans', 19: 'pigeonpeas',
    20: 'kidneybeans', 21: 'chickpea', 22: 'coffee'
}

# Fit Label Encoders
def fit_label_encoders(df, column_name, additional_values=[]):
    le = LabelEncoder()
    unique_values = list(df[column_name].unique()) + additional_values
    le.fit(unique_values)
    df[column_name] = le.transform(df[column_name])
    pickle.dump(le, open(f'{column_name}_encoder.pkl', 'wb'))
    return le

state_encoder = fit_label_encoders(df, 'state', ['Uttar Pradesh', 'Karnataka'])
district_encoder = fit_label_encoders(df, 'district', ['Basti', 'Shimoga'])
market_encoder = fit_label_encoders(df, 'market', ['Local Market', 'Shimoga Market'])
crop_name_encoder = fit_label_encoders(df, 'crop_name', ['Wheat'])

# Recommendation Function
def recommendation(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    features_scaled = mx.transform(features)
    features_standardized = sc.transform(features_scaled)
    prediction = randclf.predict(features_standardized)
    crop_name = crop_dict.get(int(prediction[0]), "Unknown crop")
    return crop_name

# Flask Routes
@app.route('/')
def home():
    return "Welcome to the Crop Recommendation API!"

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    if not all(key in data for key in required_fields):
        return jsonify({'error': 'Missing input data'}), 400

    prediction = recommendation(
        data['N'], data['P'], data['K'], 
        data['temperature'], data['humidity'], 
        data['ph'], data['rainfall']
    )

    return jsonify({'predicted_crop': prediction})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        new_data = pd.DataFrame({
            'state': [data['state']],
            'district': [data['district']],
            'market': [data['market']],
            'crop_name': [data['crop_name']],
            'min_price': [float(data['min_price'])],
            'max_price': [float(data['max_price'])]
        })

        # Encoding function
        def encode_column(column_name, encoder):
            try:
                return encoder.transform(new_data[column_name])
            except ValueError:
                unique_values = list(encoder.classes_) + list(new_data[column_name].unique())
                encoder.classes_ = np.array(unique_values)
                return encoder.transform(new_data[column_name])
        
        new_data['state'] = encode_column('state', state_encoder)
        new_data['district'] = encode_column('district', district_encoder)
        new_data['market'] = encode_column('market', market_encoder)
        new_data['crop_name'] = encode_column('crop_name', crop_name_encoder)

        # Predictions
        predicted_price_xgb = float(xgb_model.predict(new_data)[0])
        predicted_price_nn = float(nn_model.predict(new_data)[0][0])

        return jsonify({
            'predicted_price_xgb': predicted_price_xgb,
            'predicted_price_nn': predicted_price_nn
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
