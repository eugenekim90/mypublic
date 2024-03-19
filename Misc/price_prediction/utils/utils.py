import requests
import pandas as pd
from datetime import datetime, timedelta
from utils.indicators import get_sma, get_rsi, get_macd_signal
from models.lstm import build_lstm_model

def technical_indicators(df):
    """
    Creating SMA_5, SMA_10, RSI and MACD indicators
    """
    df['sma_5'] = get_sma(df['price'], window=5)
    df['sma_10'] = get_sma(df['price'], window=10)
    df['rsi'] = get_rsi(df['price'])
    df['macd'], df['signal'], df['macd_hist'] = get_macd_signal(df['price'])
    return df

def create_seq_features(df, columns, window):
    """
    Creating Sequence Features for LSTM model
    """
    df = df.copy()
    for column in columns:
        for i in range(1, window+1):
            df[f'{column}(t-{i})'] = df[column].shift(i)
    df.dropna(inplace=True)
    return df

def preprocess_data(df, window):
    init_price = df['price'].iloc[0]
    df['price'] = df['price'] / init_price

    df = technical_indicators(df)

    columns_to_seq = ['price', 'sma_5', 'sma_10', 'rsi', 'macd', 'signal']

    f_df = create_seq_features(df, columns_to_seq, window)
    features = f_df.drop(columns=columns_to_seq + ['time', 'signal' , 'macd_hist'])[:-1] # Using only past data
    targets = f_df['price'].shift(-1).dropna() # Predicting the next time interval price
    return features, targets


# Fetch real-time data
def fetch_historical_data(symbol="BTC-USD", granularity=300):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles?granularity={granularity}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data from API. Status code: {response.status_code}")
    
    df = pd.DataFrame(response.json(), columns=["time", "low", "high", "open", "close", "volume"])

    df['time'] = df['time'].apply(lambda x: pd.to_datetime(datetime.fromtimestamp(x)))
    df.sort_values('time', inplace=True)
    df = df[['time', 'close']]
    df.rename(columns={'close': 'price'}, inplace=True)
    return df

def fetch_real_time_latest_data(symbol="BTC-USD", granularity=300):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles?granularity={granularity}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data from API. Status code: {response.status_code}")
    latest_data = response.json()[0]

    df_latest = pd.DataFrame([latest_data], columns=["time", "low", "high", "open", "close", "volume"])

    df_latest['time'] = df_latest['time'].apply(lambda x: pd.to_datetime(datetime.fromtimestamp(x)))
    df_latest = df_latest[['time', 'close']]
    df_latest.rename(columns={'close': 'price'}, inplace=True)
    return df_latest
