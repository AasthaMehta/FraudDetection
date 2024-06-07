from flask import Flask, render_template, request
import pandas as pd
from check import FraudDetectionModel

app = Flask(__name__)

# Initialize the model
model = FraudDetectionModel()
model.load_data('path_to_train_data.csv', 'path_to_test_data.csv')  # Provide the correct paths
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
