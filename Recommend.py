import numpy as np
import pickle
import logging
import os
import gc
from flask import Flask, request, jsonify
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from memory_profiler import profile

app = Flask(__name__)

# ✅ Configure logging
logging.basicConfig(level=logging.INFO)

# ✅ Define file paths
MODEL_DIR = 'Model'  # Adjust if needed
minmax_path = os.path.join(MODEL_DIR, 'minmaxscaler.pkl')
stand_path = os.path.join(MODEL_DIR, 'standscaler.pkl')
model_path = os.path.join(MODEL_DIR, 'model.pkl')

# ✅ Load Pre-trained Scalers & Model
with open(minmax_path, 'rb') as minmax_file:
    minmax_scaler = pickle.load(minmax_file)

with open(stand_path, 'rb') as stand_file:
    standard_scaler = pickle.load(stand_file)

with open(model_path, 'rb') as model_file:
    randclf = pickle.load(model_file)  # RandomForestClassifier for Crop Recommendation

# ✅ Crop Dictionary
crop_dict = {
    1: 'rice', 2: 'maize', 3: 'jute', 4: 'cotton', 5: 'coconut',
    6: 'papaya', 7: 'orange', 8: 'apple', 9: 'muskmelon', 10: 'watermelon',
    11: 'grapes', 12: 'mango', 13: 'banana', 14: 'pomegranate', 15: 'lentil',
    16: 'blackgram', 17: 'mungbean', 18: 'mothbeans', 19: 'pigeonpeas',
    20: 'kidneybeans', 21: 'chickpea', 22: 'coffee'
}

# ✅ Crop Recommendation Function (7 Features)
def recommend_crop(N, P, K, temperature, humidity, ph, rainfall):
    try:
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        # Scale data
        features_scaled = minmax_scaler.transform(features)
        features_standardized = standard_scaler.transform(features_scaled)

        # Predict crop
        prediction = randclf.predict(features_standardized)
        predicted_class = int(prediction[0])
        crop_name = crop_dict.get(predicted_class, "Unknown crop")

        logging.info(f"Recommendation: {crop_name}")

        return crop_name
    except Exception as e:
        logging.error(f"Error in recommendation: {str(e)}")
        return "Error in prediction"

# ✅ Flask Routes
@app.route('/')
def home():
    return "Welcome to the Crop Recommendation API!"

@app.route('/recommend', methods=['POST'])
@profile
def recommend():
    try:
        data = request.get_json()
        logging.info(f"Received data: {data}")

        required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing input data'}), 400

        # Extract values
        N = float(data['N'])
        P = float(data['P'])
        K = float(data['K'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])

        # Get recommendation
        prediction = recommend_crop(N, P, K, temperature, humidity, ph, rainfall)

        # Clean up memory
        gc.collect()

        return jsonify({'predicted_crop': prediction})

    except ValueError as ve:
        logging.error(f"ValueError: {str(ve)}")
        return jsonify({'error': f'ValueError: {str(ve)}'}), 400
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

# ✅ Run Flask API
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logging.info(f"Running on port: {port}")
    app.run(host="0.0.0.0", port=port)
