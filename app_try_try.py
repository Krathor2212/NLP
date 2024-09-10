from flask import Flask, request, jsonify, redirect, url_for
import joblib
import pandas as pd
import os
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Global variable to track the model's status
model_status = {
    'status': 'Not started',  # Initial status
    'message': ''
}

# Directory and path where the model will be saved
model_dir = os.path.join(os.getcwd(), 'model')
model_path = os.path.join(model_dir, 'classifier.pkl')

def download_model():
    global model_status
    model_status['status'] = 'Downloading'
    try:
        os.makedirs(model_dir, exist_ok=True)  # Ensure model directory exists
        
        # Replace with your OneDrive direct download link
        model_url = 'https://annauniv0-my.sharepoint.com/:u:/g/personal/2022115109_student_annauniv_edu/EY_JBM5lZ-5AjHnqDikLmXcBX51tyysyP-uyI3YnSvC2_g?e=B08gk6'
        
        # Download the file from OneDrive
        response = requests.get(model_url)
        if response.status_code == 200:
            with open(model_path, 'wb') as f:
                f.write(response.content)
            model_status['status'] = 'Downloaded'
            model_status['message'] = 'Model successfully downloaded.'
        else:
            raise Exception("Download failed with status code: {}".format(response.status_code))
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

@app.route('/analyze', methods=['POST'])
def analyze():
    if model_status['status'] != 'Ready':
        return jsonify({'error': 'Model is not ready', 'details': model_status}), 503

    data = request.get_json()
    message = data.get('message', '')

    if message:
        review = pd.Series([message])  # Wrap the message in a list
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
