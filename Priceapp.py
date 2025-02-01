import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import logging
import os
from sklearn.utils.validation import check_array
from memory_profiler import profile
import gc

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

def load_and_prepare_crop_data():
    """Load and prepare the crop recommendation dataset"""
    crop_file_path = os.path.join('Model', 'Crop_recommendation.csv')
    crop_df = pd.read_csv(crop_file_path)
    return crop_df

def load_and_prepare_price_data():
    """Load and prepare the price prediction dataset"""
    price_file_path = os.path.join('Model', 'Cropprice.csv')
    price_df = pd.read_csv(price_file_path)
    price_df.ffill(inplace=True)
    return price_df

# Load datasets
crop_df = load_and_prepare_crop_data()
price_df = load_and_prepare_price_data()

# Define features for price prediction
PRICE_FEATURES = ['state', 'district', 'market', 'crop_name', 'min_price', 'max_price']

# Crop dictionary for recommendation
crop_dict = {
    1: 'rice', 2: 'maize', 3: 'jute', 4: 'cotton', 5: 'coconut',
    6: 'papaya', 7: 'orange', 8: 'apple', 9: 'muskmelon', 10: 'watermelon',
    11: 'grapes', 12: 'mango', 13: 'banana', 14: 'pomegranate', 15: 'lentil',
    16: 'blackgram', 17: 'mungbean', 18: 'mothbeans', 19: 'pigeonpeas',
    20: 'kidneybeans', 21: 'chickpea', 22: 'coffee'
}

# Initialize and train recommendation models
def train_recommendation_model(crop_df):
    """Train the crop recommendation model"""
    X = crop_df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = crop_df['label']
    
    # Create and fit scalers
    mx = MinMaxScaler()
    sc = StandardScaler()
    X_scaled = mx.fit_transform(X)
    X_standardized = sc.fit_transform(X_scaled)
    
    # Train Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_standardized, y)
    
    return rf_model, mx, sc

# Initialize and train price prediction models
def train_price_prediction_models(price_df):
    """Train the price prediction models"""
    # Fit Label Encoders
    encoders = {}
    for col in ['state', 'district', 'market', 'crop_name']:
        le = LabelEncoder()
        price_df[col] = le.fit_transform(price_df[col])
        encoders[col] = le
    
    X = price_df[PRICE_FEATURES]
    y = price_df['suggested_price']
    
    # Create and fit scalers
    mx = MinMaxScaler()
    sc = StandardScaler()
    X_scaled = mx.fit_transform(X)
    X_standardized = sc.fit_transform(X_scaled)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.2, random_state=42)
    
    # Train XGBoost model
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)
    xgb_model.fit(X_train, y_train)
    
    # Train Neural Network model
    nn_model = Sequential([
        Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    nn_model.compile(optimizer='adam', loss='mean_squared_error')
    nn_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
    
    return xgb_model, nn_model, encoders, mx, sc

# Train models
rf_model, crop_mx, crop_sc = train_recommendation_model(crop_df)
xgb_model, nn_model, price_encoders, price_mx, price_sc = train_price_prediction_models(price_df)

def recommend_crop(N, P, K, temperature, humidity, ph, rainfall):
    """Recommend crop based on soil and climate parameters"""
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    features_scaled = crop_mx.transform(features)
    features_standardized = crop_sc.transform(features_scaled)
    prediction = rf_model.predict(features_standardized)
    predicted_class = int(prediction[0])
    crop_name = crop_dict.get(predicted_class, "Unknown crop")
    return crop_name

@app.route('/')
def home():
    return "Welcome to the Crop Recommendation and Price Prediction API!"

@app.route('/recommend', methods=['POST'])
def recommend():
    """Endpoint for crop recommendation"""
    try:
        data = request.get_json()
        required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        prediction = recommend_crop(
            data['N'], data['P'], data['K'],
            data['temperature'], data['humidity'],
            data['ph'], data['rainfall']
        )
        
        return jsonify({'recommended_crop': prediction})
    except Exception as e:
        logging.error("Error in recommendation: %s", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
@profile
def predict():
    """Endpoint for price prediction"""
    try:
        data = request.get_json()
        logging.info("Received Data: %s", data)
        
        # Create DataFrame with required features
        new_data = pd.DataFrame({
            'state': [data['state']],
            'district': [data['district']],
            'market': [data['market']],
            'crop_name': [data['crop_name']],
            'min_price': [float(data['min_price'])],
            'max_price': [float(data['max_price'])]
        })
        
        # Encode categorical variables
        for col in ['state', 'district', 'market', 'crop_name']:
            new_data[col] = price_encoders[col].transform(new_data[col])
        
        # Scale features
        new_data_scaled = price_mx.transform(new_data)
        new_data_standardized = price_sc.transform(new_data_scaled)
        
        # Make predictions
        predicted_price_xgb = float(xgb_model.predict(new_data_standardized)[0])
        predicted_price_nn = float(nn_model.predict(new_data_standardized)[0][0])
        
        gc.collect()  # Run garbage collection
        
        return jsonify({
            'predicted_price_xgb': predicted_price_xgb,
            'predicted_price_nn': predicted_price_nn
        })
    except Exception as e:
        logging.error("Error in prediction: %s", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logging.info(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port)
