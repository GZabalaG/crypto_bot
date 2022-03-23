# Utils methods for this project
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame

class FeaturesExtractor:

    def __init__(self): # Constructor
        pass

    def open_close(self, row):
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
        #Calculate the On Balance Volume (OBV)
        obv = []
        obv.append(0)

        #Loop through the data set (close price) from the second row (index 1) to the end of the data set
        for i in range(1, len(df['close'])):
            j = i-1
            if df.iloc[i].loc['close'] > df.iloc[j].loc['close']:
                obv.append(obv[-1] + df['Volume USDT'].iloc[i])
            elif df.iloc[i].loc['close'] < df.iloc[j].loc['close']:
                obv.append(obv[-1] - df['Volume USDT'].iloc[i])
            else:
                obv.append(obv[-1])
        return obv

    def get_obv_diff(self, df):
        return df['OBV'].diff().fillna(df['OBV'])

    def get_obv_signal(self, obv, period):
        '''
        Returns the mean of differences in the past periods
        '''
        signal = obv.rolling(period).mean()
        signal_norm=(signal-signal.min())/(signal.max()-signal.min())
        return signal_norm

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

    def get_shift(self, df, column, shift):
        return df[column].shift(periods = shift)

    def get_operation(self, close, pbuy, psell):
        '''
        Creates artificial column with bests operations to make in each point

        pbuy: indicator of how much does the value has to grow to consider it inflexion buy point
        psell: indicator of how much the value has to decrease to consider it inflexion sell point (aprox 5% over last higher point)
        Maybe done with api (spread)

        Iteration over close prices:
        Set ref point to calculate condition over pbuy or psell (could be last high or last low)
        Set first value to hold
        If value met the condition of pbuy or psell set the previous point to buy or sell (check not to make two continuous buys or sells)
        If not set to hold

        Operations can be:
            buy
            sell
            hold
        '''
        op = []
        prev_val = close.iloc[0]
        prev_state = 'eq'
        for i, value in close.items():
            if value > prev_val:
                state = 'asc'
            elif prev_val > value:
                state = 'desc'
            else:
                state = 'eq'

            if prev_state == state: 
                ops = 'hold'
                op.append('hold')
            ## Chequear que el valor baje el psell indicado. 
            # Se debe iterar sobre el dataset close a partir del valor actual para ver si en algun momento bajamos de psell antes de volver a subir
            # Si no bajamos de psell no se vende y se indica hold
            elif (prev_state == 'asc' or prev_state == 'eq') and state == 'desc':
                sell_flag = False
                next_closes = close.loc[i:].copy()
                prev_value_2 = close.loc[i-1]
                for i_2, value_2 in next_closes.items():
                    if value_2 > prev_value_2:
                        break
                    if (prev_val-value_2)/prev_val > psell:
                        op.append('sell')
                        ops = 'sell'
                        sell_flag = True
                        break
                if not sell_flag: 
                    op.append('hold')
                    ops = 'hold'
                
                #ops = 'sell'
                #op.append('sell')
            # Idem for buy
            elif (prev_state == 'desc' or prev_state == 'eq') and state == 'asc':
                sell_flag = False
                next_closes = close.loc[i:].copy()
                prev_value_2 = close.loc[i-1]
                for i_2, value_2 in next_closes.items():
                    if value_2 < prev_value_2:
                        break
                    if abs((prev_val-value_2)/prev_val) > pbuy:
                        op.append('buy')
                        ops = 'buy'
                        sell_flag = True
                        break
                if not sell_flag: 
                    op.append('hold')
                    ops = 'hold'
                
                #ops = 'buy'
                #op.append('buy')
            else:
                op.append('hold')


            prev_val = close[i]
            prev_state = state
        op = op[1:]
        op.append('hold')
        return op

    def get_close_diff(self, close, period):
        '''
        Returns column with diff between close prices separated period periods
        '''

        return close - close.shift(period)