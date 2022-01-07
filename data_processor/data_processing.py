#Class with methods for data loading, cleaning, feature extraction and anomaly detection from source
#Future methods to load streaming data and upload live forecast

import pandas as pd
from utils.crypto_utils import Features


class DataProcessor:

    def __init__(self, cryptos): # Constructor
        '''
        API conector and cryptos list to analize
        '''
        self.cryptos_names = cryptos

    def load_data(self): # Load method
        '''
        Get Data from cryptos selected in constructor or requested by args
        '''
        self.crypto_df = []
        for crypto_name in self.cryptos_names:
            print('Loading...', crypto_name)
            path = '../Datasets/' + crypto_name + '.csv'
            df = pd.read_csv(path, header=[1])
            self.crypto_df.append(df)

    def clean_data(self, cryptos_names): # Clean method
        '''
        Cleans data to prepare it for better models comprehension and feature extraction
        '''
        i = 0
        for df in self.crypto_df:
            if self.cryptos_names[i] in cryptos_names: # if crypto is already loaded
                print('Drop columns')
                df.drop(columns=['symbol', 'unix', 'Volume USDT'], inplace = True)
                print('Drop Nan')
                df.dropna(inplace =True)
                print('Change date format')
                df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
                print('Reverse order')
                df = df.iloc[::-1]
            i+=1


    def get_data(self, crypto_name):
        '''
        Returns crypto dataframe with crypto_name
        '''
        return(self.crypto_df[self.cryptos_names.index(crypto_name)])

    def feature_extraction(self, cryptos_names): # Feature extraction method
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
        - Fib. Retracement
        - Exponential moving average (EMA)
        - On Balance Volume
        - SMA
        '''
        i = 0
        for df in self.crypto_df:
            if self.cryptos_names[i] in cryptos_names:
                df['High Low Difference'] = df['high'] - df['low']
                df['Open Close Difference'] = df['open'] - df['close']
                df['Result'] = df.apply(lambda row: Features.high_low(row), axis=1)
                df['Sup 5'] = Features.get_support_resistance(df, 5, 'S')
                df['Sup 50'] = Features.get_support_resistance(df, 50, 'S')
                df['Sup 200'] = Features.get_support_resistance(df, 200, 'S')
                df['Sup 500'] = Features.get_support_resistance(df, 500, 'S',)
                df['Res 5'] = Features.get_support_resistance(df, 5, 'R')
                df['Res 50'] = Features.get_support_resistance(df, 50, 'R')
                df['Res 200'] = Features.get_support_resistance(df, 200, 'R')
                df['Res 500'] = Features.get_support_resistance(df, 500, 'R')
                df['RSI'] = Features.get_rsi(df)
                df['plus_di'] = Features.get_adx(df, 14)[0]
                df['minus_di'] = Features.get_adx(df, 14)[1]
                df['ADX'] = Features.get_adx(df, 14)[2]
                df['SMA'] = Features.get_sma(df, 14)
                df['Standard Deviation'] = 0
                df['Bollinger'] = 0
                df['Stochastic Oscillator'] = 0
                df['MACD'] = 0
                df['EMA'] = 0
                df['OBV'] = 0
                df['Ichimoku'] = 0
                df['Fib Retracement'] = 0
            i+=1

    def feature_selection(self, crypto_name): # Feature selection method
        '''
        Eliminates features not relevant or highly correlated to others
        '''

    def detect_anomalies(self, crypto_name): # Anomaly detection method
        '''
        Build non supervised methods to detect outliers and anomalies and discard data
        '''

    def train_test_split(self, crypto_name): # Train-test split method
        '''
        Analize best train-test split method and returns datasets separated in train and test
        '''