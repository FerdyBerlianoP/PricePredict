from flask import Flask, request, jsonify
import requests
import os
from new_predict import prepare_and_train_model, predict_next_price

app = Flask(__name__)

API_KEY = 'JW3KVVQ94FV09N81'
ALPHA_VANTAGE_URL = 'https://www.alphavantage.co/query'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symbol = data.get('symbol')
    
    if not symbol:
        return jsonify({'error': 'Symbol is required'}), 400

    try:
        url = f"{ALPHA_VANTAGE_URL}?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=15min&apikey={API_KEY}"
        response = requests.get(url)
        stock_data = response.json()

        if 'Time Series (15min)' not in stock_data:
            return jsonify({'error': 'Time series data not found for the given symbol'}), 404
        
        last_close, accuracy = prepare_and_train_model(stock_data)
        next_price = predict_next_price(last_close)

        return jsonify({
            'last_close': last_close,
            'predicted_next_price': next_price
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'running'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

