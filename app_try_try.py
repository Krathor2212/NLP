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
model_path = os.path.join(os.getcwd(), 'model', 'classifier.pkl')

def download_model():
    global model_status
    try:
        model_status['status'] = 'Downloading'
        model_url = 'https://drive.google.com/file/d/1Nshs0xEK-5XAzL8shnKzpd5mldz2WfF9/view?usp=sharing'
        gdown.download(model_url, model_path, quiet=False)
        model_status['status'] = 'Downloaded'
        model_status['message'] = 'Model successfully downloaded.'
    except Exception as e:
        model_status['status'] = 'Error'
        model_status['message'] = f'Error downloading model: {str(e)}'

# Check if the model already exists locally
if not os.path.exists(model_path):
    download_model()
else:
    model_status['status'] = 'Loaded'
    model_status['message'] = 'Model already exists locally.'

try:
    # Load the model
    classifier = joblib.load(model_path)
    model_status['status'] = 'Ready'
    model_status['message'] = 'Model loaded successfully.'
except Exception as e:
    model_status['status'] = 'Error'
    model_status['message'] = f'Error loading model: {str(e)}'

def predictfunc(reviews):
    predictions = classifier.predict(reviews)
    predict = int(predictions[0])
    return predict

@app.route('/')
def index():
    # Redirect to the model status page or return a simple welcome message
    return redirect(url_for('get_model_status'))
    # Alternatively, return a simple HTML message
    # return "<h1>Welcome to the Sentiment Analysis API</h1>"

@app.route('/analyze', methods=['POST'])
def analyze():
    if model_status['status'] != 'Ready':
        return jsonify({'error': 'Model is not ready', 'details': model_status}), 503

    data = request.get_json()
    message = data.get('message', '')

    if message:
        review = pd.Series(message)
        prediction = predictfunc(review)
        is_bully = prediction == 0
        print("bully", is_bully)
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
