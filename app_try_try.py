from flask import Flask, request, jsonify, redirect, url_for
import joblib
import pandas as pd
import os
import gdown
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Global variable to track the model's status
model_status = {
    'status': 'Not started',  # Initial status
    'message': ''
}

# Path where the model will be saved
model_dir = os.path.join(os.getcwd(), 'model')
model_path = os.path.join(model_dir, 'classifier.pkl')

def download_model():
    global model_status
    model_status['status'] = 'Downloading'
    try:
        os.makedirs(model_dir, exist_ok=True)  # Ensure model directory exists
        model_url = 'https://drive.google.com/uc?export=download&id=1Nshs0xEK-5XAzL8shnKzpd5mldz2WfF9'
        gdown.download(model_url, model_path, quiet=False)
        model_status['status'] = 'Downloaded'
        model_status['message'] = 'Model successfully downloaded.'
    except Exception as e:
        model_status['status'] = 'Error'
        model_status['message'] = f'Error downloading model: {str(e)}'

def load_model():
    global model_status
    try:
        global classifier
        classifier = joblib.load(model_path)
        model_status['status'] = 'Ready'
        model_status['message'] = 'Model loaded successfully.'
    except Exception as e:
        model_status['status'] = 'Error'
        model_status['message'] = f'Error loading model: {str(e)}'

# Check if the model already exists locally
if not os.path.exists(model_path):
    download_model()
load_model()

def predict_func(reviews):
    predictions = classifier.predict(reviews)
    return int(predictions[0])

@app.route('/')
def index():
    return redirect(url_for('get_model_status'))

@app.route('/analyze', methods=['POST'])
def analyze():
    if model_status['status'] != 'Ready':
        return jsonify({'error': 'Model is not ready', 'details': model_status}), 503

    data = request.get_json()
    message = data.get('message', '')

    if message:
        review = pd.Series(message)
        prediction = predict_func(review)
        is_bully = prediction == 0
        response = {
            'prediction': prediction,
            'is_bullying': is_bully
        }
        return jsonify(response)
    else:
        return jsonify({'error': 'No message provided'}), 400

@app.route('/model_status', methods=['GET'])
def get_model_status():
    return jsonify(model_status)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
