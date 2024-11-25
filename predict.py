import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

API_KEY = '3T44KJ0WW36CBINK'

symbol = 'MSFT'

url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=15min&apikey={API_KEY}'

response = requests.get(url)
data = response.json()

if 'Time Series (15min)' not in data:
    print("Error fetching data from Alpha Vantage API")
    exit()

time_series = data['Time Series (15min)']

df = pd.DataFrame.from_dict(time_series, orient='index')
df = df[['4. close']]
df.columns = ['close']
df.index = pd.to_datetime(df.index)

df = df.reset_index()
df['Prev_Close'] = df['close'].shift(1)
df = df.dropna()
# print(df.head())

X = df[['Prev_Close']]
y = df['close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
# print(f'Model accuracy: {model.score(X_test, y_test)}')

joblib.dump(model, 'stock_predictor_model.pkl')
