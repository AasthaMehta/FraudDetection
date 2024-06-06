from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
with open('trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        features = [float(data['amt']), data['city'], data['state'], data['zip']]
        # Process other features as needed
        prediction = model.predict([features])[0]
        return render_template('result.html', prediction=prediction)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

import os
os.chmod('app.py', 0o755)
