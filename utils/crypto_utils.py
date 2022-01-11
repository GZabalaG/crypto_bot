# Utils methods for this project
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame

class FeaturesExtractor:

    def __init__(self): # Constructor
        pass

    def high_low(self, row):
        '''
        Returns the period high-low difference result
        '''
        return 1 if row['Open Close Difference'] > 0 else 0

    def get_support_resistance(self, df, shift, op):
        '''
        Returns the support or resistance level of shift periods before based on maximum or minimum
        '''
        if op == 'S':
            return pd.DataFrame(df['close'].rolling(window = shift).min())
        elif op == 'R':
            return pd.DataFrame(df['close'].rolling(window = shift).max())
        else: 
            print('Error: not valid operation. R: resistance, S: support')

    def get_rsi(self, df, periods = 14, ema = True):
        """
        Returns the relative strength index.
        """
        close_delta = df['close'].diff()

        # Make two series: one for lower closes and one for higher closes
        up = close_delta.clip(lower=0)
        down = -1 * close_delta.clip(upper=0)
        
        if ema == True:
            # Use exponential moving average
            ma_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
            ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
        else:
            # Use simple moving average
            ma_up = up.rolling(window = periods, adjust=False).mean()
            ma_down = down.rolling(window = periods, adjust=False).mean()
            
        rsi = ma_up / ma_down
        rsi = 100 - (100/(1 + rsi))
        return rsi

    def get_adx(self, df, lookback):
        '''
        Return plus_di, minus_di and ADX in lookback periods
        '''
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr1 = pd.DataFrame(df['high'] - df['low'])
        tr2 = pd.DataFrame(abs(df['high'] - df['close'].shift(1)))
        tr3 = pd.DataFrame(abs(df['low'] - df['close'].shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
        atr = tr.rolling(lookback).mean()
        
        plus_di = 100 * (plus_dm.ewm(alpha = 1/lookback).mean() / atr)
        minus_di = abs(100 * (minus_dm.ewm(alpha = 1/lookback).mean() / atr))
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        adx = ((dx.shift(1) * (lookback - 1)) + dx) / lookback
        adx_smooth = adx.ewm(alpha = 1/lookback).mean()
        
        return plus_di, minus_di, adx_smooth

    def get_sma(self, df, n):
        '''
        Returns SMA of n periods
        '''
        return pd.DataFrame(df['close'].rolling(window = n).mean())
    
    def get_std(self, df, n):
        '''
        Returns Standard deviation in n rolling periods
        '''
        return pd.DataFrame(df['close'].rolling(window = n).std())

    def get_bollinger_bands(self, df, n, factor):
        '''
        Returns lower and upper bollinger bands
        '''
        sma = self.get_sma(df, n)
        std = self.get_std(df, n)
        upper_band = sma + (factor * std)
        lower_band = sma - (factor * std)
        return upper_band, lower_band
    
    def get_stoch(self, df, k_period, d_period):
        high = df['high'].rolling(k_period).max()
        low = df['low'].rolling(k_period).min()
        so_k = (df['close'] - low)*100/(high - low)
        so_d = so_k.rolling(d_period).mean()
        return so_k, so_d
    
    def get_macd(self, df, slow, fast, smooth):
        exp1 = df['close'].ewm(span = fast, adjust = False).mean()
        exp2 = df['close'].ewm(span = slow, adjust = False).mean()
        macd = pd.DataFrame(exp1 - exp2).rename(columns = {'close':'macd'})
        signal = pd.DataFrame(macd.ewm(span = smooth, adjust = False).mean()).rename(columns = {'macd':'signal'})
        hist = pd.DataFrame(macd['macd'] - signal['signal']).rename(columns = {0:'hist'})
        return macd, signal, hist

    def get_ema(self, df, period):
        return df['close'].ewm(span = period, adjust = False).mean()

    def get_obv(self, df):
        return np.sign(df['close'].diff()) * df['Volume USDT'].fillna(0).cumsum()

    def get_ichimoku(self, df):
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2))
        period9_high = df['high'].rolling(window = 9).max()
        period9_low = df['low'].rolling(window=9).min()
        tenkan_sen = (period9_high + period9_low) / 2

        # Kijun-sen (Base Line): (26-period high + 26-period low)/2))
        period26_high = df['high'].rolling(window = 26).max()
        period26_low = df['low'].rolling(window = 26).min()
        kijun_sen = (period26_high + period26_low) / 2

        # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2))
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2))
        period52_high = df['high'].rolling(window = 52).max()
        period52_low = df['low'].rolling(window = 52).max()
        senkou_span_b = ((period52_high + period52_low) / 2).shift(26)

        # The most current closing price plotted 22 time periods behind (optional)
        chikou_span = df['close'].shift(-22) # 22 according to investopedia

        return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span
    
    def get_atr(self, df, period):
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)

        return true_range.rolling(period).sum()/period