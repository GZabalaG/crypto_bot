# Utils methods for this project
import pandas as pd
from pandas.core.frame import DataFrame

class Features:

    def __init__(self): # Constructor
        pass

    def high_low(row):
        '''
        Returns the period high-low difference result
        '''
        return 1 if row['Open Close Difference'] > 0 else 0

    def get_support_resistance(df, shift, op):
        '''
        Returns the support or resistance level of shift periods before based on maximum or minimum
        '''
        if op == 'S':
            return pd.DataFrame(df['close'].rolling(window = shift).min())
        elif op == 'R':
            return pd.DataFrame(df['close'].rolling(window = shift).max())
        else: 
            print('Error: not valid operation. R: resistance, S: support')

    def get_rsi(df, periods = 14, ema = True):
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

    def get_adx(df, lookback):
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

    def get_sma(df, n):
        '''
        Return SMA of n periods
        '''
        return pd.DataFrame(df['close'].rolling(window = n).mean())

    