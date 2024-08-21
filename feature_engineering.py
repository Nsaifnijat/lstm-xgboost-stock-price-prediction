import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import pandas_ta as ta
from sklearn import preprocessing

import MetaTrader5 as mt
import datetime as dt
from statsmodels.tsa.seasonal import seasonal_decompose

mt.initialize()
def data_download(symbol):
    current_time  = dt.datetime.now()+dt.timedelta(days=2) 
    timeframe = mt.TIMEFRAME_H1
    df = pd.DataFrame(mt.copy_rates_from(symbol, timeframe, current_time, 50000))
   
    df.columns = ['time','open','high','low','close','volume','spread','real_volume']
    df['time'] = pd.to_datetime(df['time'],unit='s')
    #df['datetime']=df['datetime'].dt.tz_localize('utc').dt.tz_convert('US/Eastern')
    df['time']=df['time'].dt.tz_localize('utc').dt.tz_convert('US/Pacific')
    df['time']=df['time'].dt.tz_localize(None)
    df = df.set_index('time')
    return df

df = data_download('EURUSD.sml')

# Add time-based features
df['hour'] = df.index.hour
df['minute'] = df.index.minute
df['day_of_week'] = df.index.dayofweek
df['day_of_month'] = df.index.day
df['month'] = df.index.month


# Slope Calculation
window = 10  # Example window size, can be adjusted
df['slope'] = ta.slope(df['close'], length=window)


for lag in [10, 20]:
    df[f'close_lag_{lag}'] = df['close'].shift(lag)


df['rolling_std_20'] = df['close'].rolling(window=20).std()

"""Volume-based Indicators: Incorporate volume-based indicators
 like Chaikin Money Flow (CMF) and Volume Price Trend (VPT)."""
df['vpt'] = (df['volume'] * (df['close'] - df['close'].shift(1)) / df['close'].shift(1)).cumsum()


decomposition = seasonal_decompose(df['close'], model='additive', period=365)
df['trend'] = decomposition.trend
df['seasonal'] = decomposition.seasonal
df['residual'] = decomposition.resid


fft = np.fft.fft(df['close'])
df['fft_real'] = np.real(fft)
df['fft_imag'] = np.imag(fft)


# Peaks and Depths for selected epochs
epochs = [5, 20, 100]
for epoch in epochs:
    peaks = argrelextrema(df['close'].values, np.greater, order=epoch)[0]
    df[f'peaks_{epoch}'] = df['close'][peaks]
    df[f'peaks_{epoch}'].ffill(inplace=True)

    depths = argrelextrema(df['close'].values, np.less, order=epoch)[0]
    df[f'depth_{epoch}'] = df['close'][depths]
    df[f'depth_{epoch}'].ffill(inplace=True)

# SMA and EMA for selected epochs
epochs = [10, 50, 200]
for epoch in epochs:
    df[f'sma_{epoch}'] = ta.sma(df['close'], length=epoch)
    #df[f'ema_{epoch}'] = ta.ema(df['close'], length=epoch)

# Technical Indicators
df['cci_20'] = ta.momentum.cci(df['high'], df['low'], df['close'], length=20)
df['rsi'] = ta.rsi(df['close'], length=10)
macd = ta.macd(df['close'])
df['macd'] = macd['MACDh_12_26_9']
df['signal'] = macd['MACDs_12_26_9']
df['atr'] = ta.atr(df['high'], df['low'], df['close'])


df['williams_r'] = ta.willr(df['high'], df['low'], df['close'])
df['obv'] = ta.obv(df['close'], df['volume'])
bands = ta.bbands(df['close'])
df['bb_width'] = bands['BBU_5_2.0'] - bands['BBL_5_2.0']
df['cmf'] = ta.cmf(df['high'], df['low'], df['close'], df['volume'])
df['bull_power'] = df['high'] - ta.ema(df['close'], length=13)
df['bear_power'] = df['low'] - ta.ema(df['close'], length=13)
df['keltner_upper'] = ta.ema(df['close'], length=20) + (ta.atr(df['high'], df['low'], df['close']) * 2)
df['keltner_lower'] = ta.ema(df['close'], length=20) - (ta.atr(df['high'], df['low'], df['close']) * 2)
df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'])
df['parabolic_sar'] = ta.psar(df['high'], df['low'], df['close'])['PSARr_0.02_0.2']
def fractals(df):
    df['fractal_high'] = df['high'][(df['high'].shift(2) < df['high']) & (df['high'].shift(1) < df['high']) & (df['high'].shift(-1) < df['high']) & (df['high'].shift(-2) < df['high'])]
    df['fractal_low'] = df['low'][(df['low'].shift(2) > df['low']) & (df['low'].shift(1) > df['low']) & (df['low'].shift(-1) > df['low']) & (df['low'].shift(-2) > df['low'])]
    df['fractal_high'].ffill(inplace=True)
    df['fractal_low'].ffill(inplace=True)
    return df

df = fractals(df)

#log_return 
df['log_return'] = ta.log_return(df['close'])
df['volatility'] = np.std(df['log_return'])

# Exponential Smoothing
alpha = 0.2  # Smoothing factor
df['Exp Smoothing'] = df['close'].ewm(alpha=alpha, adjust=False).mean()

# Save to CSV
df.to_csv('rawfeatures.csv')
print(df.columns)