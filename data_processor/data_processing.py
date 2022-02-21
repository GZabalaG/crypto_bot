#Class with methods for data loading, cleaning, feature extraction and anomaly detection from source
#Future methods to load streaming data and upload live forecast

import pandas as pd
from pandas.core.frame import DataFrame
import datetime
from utils.crypto_utils import FeaturesExtractor


class DataProcessor:

    def __init__(self, cryptos): # Constructor
        '''
        API conector and cryptos list to analize
        '''
        self.cryptos_names = cryptos
        self.fe = FeaturesExtractor() 

    def load_data(self): # Load method
        '''
        Get Data from cryptos selected in constructor or requested by args
        '''
        self.crypto_df = []
        for crypto_name in self.cryptos_names:
            print('Loading...', crypto_name)
            path = '../Datasets/' + crypto_name + '.csv'
            df = pd.read_csv(path, header=[1])
            #print('Reversing order')
            df = df.iloc[::-1].reset_index(drop=True)
            self.crypto_df.append(df)

    def clean_data(self, cryptos_name): # Clean method
        '''
        Cleans data to prepare it for better models comprehension and feature extraction
        '''
        i = 0
        for df in self.crypto_df:
            if self.cryptos_names[i] == cryptos_name: # if crypto is already loaded
                #print('Dropping columns')
                df.drop(columns=['symbol', 'unix'], inplace = True)
                df.drop(df.columns[5], axis=1, inplace = True)
                #print('Dropping Nan')
                df.dropna(inplace =True)
                #print('Changing date format')
                df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
            i+=1

    def get_data(self, crypto_name):
        '''
        Returns crypto dataframe with crypto_name
        '''
        return(self.crypto_df[self.cryptos_names.index(crypto_name)])

    def feature_extraction(self, cryptos_name): # Feature extraction method
        '''
        Extracts features from df
        - Close-open difference
        - Close price above or below open (boolean)
        - Support and resistance levels
        - Relative strength index (RSI)
        - Average directional index (ADX)
        - Ichimoku cloud
        - Standard deviation
        - Bollinger bands
        - Stochastic oscillator
        - Moving average convergence divergence (MACD)
        - Exponential moving average (EMA)
        - On Balance Volume
        - SMA
        '''

        i = 0
        for df in self.crypto_df:
            if self.cryptos_names[i] == cryptos_name:
                df['High Low Difference'] = df['high'] - df['low']
                df['Open Close Difference'] = df['close'] - df['open']
                df['Sup 5'] = self.fe.get_support_resistance(df, 5, 'S')
                df['Sup 50'] = self.fe.get_support_resistance(df, 50, 'S')
                df['Sup 200'] = self.fe.get_support_resistance(df, 200, 'S')
                df['Sup 500'] = self.fe.get_support_resistance(df, 500, 'S',)
                df['Res 5'] = self.fe.get_support_resistance(df, 5, 'R')
                df['Res 50'] = self.fe.get_support_resistance(df, 50, 'R')
                df['Res 200'] = self.fe.get_support_resistance(df, 200, 'R')
                df['Res 500'] = self.fe.get_support_resistance(df, 500, 'R')
                df['SMA 50'] = self.fe.get_sma(df, 50)
                df['SMA 200'] = self.fe.get_sma(df, 200)
                df['upper_b_band'] = self.fe.get_bollinger_bands(df, 14, 2)[0]
                df['lower_b_band'] = self.fe.get_bollinger_bands(df, 14, 2)[1]
                df['EMA_50'] = self.fe.get_ema(df, 50)
                df['EMA_200'] = self.fe.get_ema(df, 200)
                df['MACD'] = self.fe.get_macd(df, 26, 12, 9)[0]
                df['MACD_signal'] = self.fe.get_macd(df, 26, 12, 9)[1]
                df['MACD_histo'] = self.fe.get_macd(df, 26, 12, 9)[2]
                df['ADX'] = self.fe.get_adx(df, 14)[2]
                df['Standard Deviation'] = self.fe.get_std(df, 14)
                df['SO_K'] = self.fe.get_stoch(df, 14, 3)[0]
                df['SO_D'] = self.fe.get_stoch(df, 14, 3)[1]
                df['RSI'] = self.fe.get_rsi(df)
                df['plus_di'] = self.fe.get_adx(df, 14)[0]
                df['minus_di'] = self.fe.get_adx(df, 14)[1]
                df['OBV'] = self.fe.get_obv(df)
                df['OBV_diff'] = self.fe.get_obv_diff(df)
                df['OBV_signal'] = self.fe.get_obv_signal(df['OBV_diff'], 20)
                df['I_tenkan_sen'] = self.fe.get_ichimoku(df)[0]
                df['I_kijun_sen'] = self.fe.get_ichimoku(df)[1]
                df['I_senkou_span_a'] = self.fe.get_ichimoku(df)[2]
                df['I_senkou_span_b'] = self.fe.get_ichimoku(df)[3]
                df['I_chikou_span'] = self.fe.get_ichimoku(df)[4]
                df['ATR'] = self.fe.get_atr(df, 14)
                df['Result'] = df.apply(lambda row: self.fe.open_close(row), axis=1)
                df['weekday_num'] = df['date'].dt.dayofweek
                df.loc[df['weekday_num'] == 0, 'weekday'] = 'mon'
                df.loc[df['weekday_num'] == 1, 'weekday'] = 'tue'
                df.loc[df['weekday_num'] == 2, 'weekday'] = 'wed'
                df.loc[df['weekday_num'] == 3, 'weekday'] = 'thu'
                df.loc[df['weekday_num'] == 4, 'weekday'] = 'fri'
                df.loc[df['weekday_num'] == 5, 'weekday'] = 'sat'
                df.loc[df['weekday_num'] == 6, 'weekday'] = 'sun'
                one_hot = pd.get_dummies(df['weekday'])
                df['mon'] = one_hot['mon']
                df['tue'] = one_hot['tue']
                df['wed'] = one_hot['wed']
                df['thu'] = one_hot['thu']
                df['fri'] = one_hot['fri']
                df['sat'] = one_hot['sat']
                df['sun'] = one_hot['sun']

            i+=1

    def feature_selection(self, crypto_name, fields): # Feature selection method
        '''
        Eliminates features not relevant or highly correlated to others
        '''
        i = 0
        for df in self.crypto_df:
            if self.cryptos_names[i] == crypto_name: # if crypto is already loaded
                print('Extracting columns columns for', crypto_name)
                return df[df.columns.intersection(fields)].copy()
            i+=1

    def lstm_processing(self, crypto_df, target, prev_periods, pred_periods): # prepares data for a lstm model
        '''
        Process data for LSTM model
        target: column to use as target.

        mode:
            0: target is not in X
            1: target is in X. We have to duplicate column as column_target (close_target, result_target...)

        prev_periods: how many periods to use as X
        pred_periods: periods to shift
        '''
        print('Proccessing and arranging columns for LSTM model')
        # Create aux columns. We'll have X columns and y column
        # X columns are X-5, X-4, X-3, X-2, X-1 and y column = X
        # For multivariate we myabe use only X1-1, X2-1, X3-1, etc and y column...

        # Shift target column by 1
        target_column = self.fe.get_shift(crypto_df, target, -pred_periods)
        X_columns = pd.DataFrame()
        X_columns.empty
        for i in reversed(range(prev_periods)):
            # Create df with prev_periods columns
            # We'll get total of prev_periods rows pivoted to columns stsrting from the actual one in the loop
            for col in crypto_df.columns:
                aux_column_name = f'{col}_{i}'
                X_col = self.fe.get_shift(crypto_df, f'{col}', i)
                X_col = pd.Series.to_frame(X_col).rename(columns={col: aux_column_name})
                X_columns = pd.concat([X_columns, X_col], axis=1)

        return pd.concat([X_columns, target_column], axis=1).iloc[prev_periods-1:-pred_periods]


    def detect_anomalies(self, crypto_name): # Anomaly detection method
        '''
        Build non supervised methods to detect outliers and anomalies and discard data
        '''