from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load('stock_predictor_model.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    latest_close = request.args.get('latest_close')

    if latest_close is None:
        return jsonify({'error': 'latest_close parameter is missing'}), 400
    try:
        latest_close = float(latest_close)
    except ValueError:
        return jsonify({'error': 'latest_close must be a valid number'}), 400

    prediction = model.predict(np.array([[latest_close]]))

    return jsonify({'predicted_price': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
