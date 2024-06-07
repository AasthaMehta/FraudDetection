from flask import Flask, render_template, request, jsonify
import pandas as pd
from check import FraudDetectionModel
import os

app = Flask(__name__)

# Initialize the model
model = FraudDetectionModel()

# Define the URLs to your smaller datasets in the GitHub repository
train_data_url = 'https://raw.githubusercontent.com/AasthaMehta/FraudDetection/main/train_data.csv'
test_data_url = 'https://raw.githubusercontent.com/AasthaMehta/FraudDetection/main/test_data.csv'

# Define the file paths where the datasets will be saved locally
train_file_path = 'small_train_data.csv'
test_file_path = 'small_test_data.csv'

# Download the data
model.download_data(train_data_url, train_file_path)
model.download_data(test_data_url, test_file_path)

# Load and train the model
model.load_data(train_file_path, test_file_path)
model.train_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        features = {
            'amt': float(data['amt']),
            'city': data['city'],
            'state': data['state'],
            'zip': data['zip']
            # Add other required features here
        }
        df = pd.DataFrame([features])
        df = pd.get_dummies(df)
        prediction = model.predict(df)[0]
        return render_template('result.html', prediction=prediction)
    except Exception as e:
        return str(e)

@app.route('/evaluate', methods=['GET'])
def evaluate():
    accuracy, conf_matrix, class_report = model.evaluate_model()
    return jsonify({
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
