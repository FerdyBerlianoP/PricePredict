import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

def prepare_and_train_model(data):
    try:
        time_series = data['Time Series (15min)']
        
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df['4. close'] = pd.to_numeric(df['4. close'], errors='coerce')
        df = df[['4. close']]
        df.columns = ['close']
        df.index = pd.to_datetime(df.index)
        
        last_close = float(df['close'].iloc[0])
        
        df = df.reset_index()
        df['Prev_Close'] = df['close'].shift(1)
        df = df.dropna()
        
        X = df[['Prev_Close']].values
        y = df['close'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        
        model_data = {
            'model': model,
            'last_close': last_close,
            'accuracy': accuracy
        }
        joblib.dump(model_data, 'stock_predictor_model.pkl')
        
        print(f"Model trained successfully!")
        print(f"Last closing price: ${last_close:.2f}")
        
        return last_close, accuracy
        
    except Exception as e:
        print(f"Error in prepare_and_train_model: {str(e)}")
        raise

def predict_next_price(last_close):
    try:
        model_data = joblib.load('stock_predictor_model.pkl')
        model = model_data['model']
        prediction = model.predict([[float(last_close)]])
        return prediction[0]
    except Exception as e:
        print(f"Error in predict_next_price: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        import requests
        
        API_KEY = '3T44KJ0WW36CBINK'
        symbol = 'MSFT'
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=15min&apikey={API_KEY}'
        
        response = requests.get(url)
        data = response.json()
        
        if 'Time Series (15min)' not in data:
            print("Error: No time series data found in API response")
            print("API Response:", data)
            exit()
        
        last_close, accuracy = prepare_and_train_model(data)
        
        next_price = predict_next_price(last_close)
        print(f"Predicted next price: ${next_price:.2f}")
        
    except Exception as e:
        print(f"Main execution error: {str(e)}")