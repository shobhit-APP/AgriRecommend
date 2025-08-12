# 🌾 AgriRecommend

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [API Documentation](#api-documentation)
- [Usage Examples](#usage-examples)
- [Model Details](#model-details)
- [Deployment](#deployment)
- [Integration](#integration)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

The Crop Recommendation Model uses machine learning to suggest the best crops based on soil nutrients, pH, temperature, humidity, and rainfall. Trained on agricultural data, it helps farmers choose suitable crops, improve yield, and reduce risk, promoting sustainable farming and better productivity through data-driven decisions.

### 🎯 Problem Statement
Modern agriculture faces challenges in:
- Optimal crop selection based on soil and climate conditions
- Lack of data-driven decision-making tools
- Limited access to agricultural expertise

### 💡 Solution
- **Intelligent crop recommendations** based on environmental parameters
- **RESTful API** for easy integration with existing systems

## 🏗️ Architecture

```
AgriRecommend/
├── Model/                          # Core ML models and data
│   ├── Crop_recommendation.csv    # Training data for crop recommendations
│   ├── crop_name_encoder.pkl     # Crop name label encoder
│   ├── minmaxscaler.pkl          # Feature scaler for price model
│   ├── model.pkl                 # Main crop recommendation model
│   ├── standscaler.pkl           # Standard scaler for features
├── Recommend.py                        # Main Flask application
├── requirements.txt              # Python dependencies
├── package.json                  # Node.js dependencies (if any)
└── README.md                     # Project documentation
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step-by-Step Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/shobhit-APP/AgriRecommend.git
   cd AgriRecommend
   ```

2. **Create and activate virtual environment**
   ```bash
   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   
   # Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import flask, sklearn, pandas, numpy; print('All dependencies installed successfully!')"
   ```

### Running the Application

1. **Navigate to the Model directory**
   ```bash
   cd Model
   ```

2. **Start the Flask server**
   ```bash
   # Development mode
   flask --app app.py --debug run
   
   # Production mode
   flask --app app.py run --host=0.0.0.0 --port=5000
   ```

3. **Access the API**
   - Local: http://127.0.0.1:5000
   - Network: http://your-ip:5000

## 📚 API Documentation

### Base URL
```
http://127.0.0.1:5000
```

### Endpoints

####  Crop Recommendation

**Endpoint:** `POST /recommend`

**Description:** Recommends the best crop based on soil and climate parameters.

**Request Body:**
```json
{
    "N": 90,           // Nitrogen content (kg/ha)
    "P": 42,           // Phosphorous content (kg/ha)
    "K": 43,           // Potassium content (kg/ha)
    "temperature": 20.87, // Temperature in Celsius
    "humidity": 82.00,    // Relative humidity (%)
    "ph": 6.50,          // Soil pH value
    "rainfall": 202.93    // Rainfall in mm
}
```

**Response:**
```json
{
    "recommended_crop": "rice",
    "success": true
}
```
```Error reponse
{
    "error": "Missing required parameter: N",
    "success": false,
    "status_code": 400
}
```

## 💻 Usage Examples

### Python Example
```python
import requests
import json

# Crop Recommendation
url = "http://127.0.0.1:5000/recommend"
data = {
    "N": 90, "P": 42, "K": 43,
    "temperature": 20.87, "humidity": 82.00,
    "ph": 6.50, "rainfall": 202.93
}

response = requests.post(url, json=data)
result = response.json()
print(f"Recommended crop: {result['recommended_crop']}")
```

### cURL Example
```bash
# Crop Recommendation
curl -X POST http://127.0.0.1:5000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "N": 90, "P": 42, "K": 43,
    "temperature": 20.87, "humidity": 82.00,
    "ph": 6.50, "rainfall": 202.93
  }'

## 🧠 Model Details

### Crop Recommendation Model
- **Algorithm**: Random Forest Classifier
- **Features**: N, P, K, temperature, humidity, pH, rainfall
- **Accuracy**: ~99% on test data
- **Training Data**: 2,200+ samples across 22 crop types


## 🌐 Deployment

### Local Development
```bash
flask --app app.py --debug run
```

### Production Deployment

#### Using Render
1. Connect your GitHub repository to Render
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `gunicorn app:app`
4. Deploy and get your live URL

#### Using Heroku
```bash
# Install Heroku CLI and login
heroku create agripredict-app
git push heroku main
heroku open
```

#### Using AWS EC2
```bash
# Install dependencies on EC2
sudo apt update
sudo apt install python3-pip nginx
pip3 install -r requirements.txt

# Use gunicorn for production
gunicorn --bind 0.0.0.0:5000 app:app
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```
## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Ensure code quality: `black . && flake8`
5. Commit changes: `git commit -m 'Add feature description'`
6. Push to branch: `git push origin feature-name`
7. Open a Pull Request

### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable names
- Add docstrings for functions
- Include unit tests for new features

### Reporting Issues
Please use the GitHub issue tracker to report bugs or request features.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/shobhit-APP/AgriPredict/issues)
- **Discussions**: [GitHub Discussions](https://github.com/shobhit-APP/AgriPredict/discussions)

---

**Built with ❤️ for farmers and agricultural innovation**

*Last updated: December 2024*
