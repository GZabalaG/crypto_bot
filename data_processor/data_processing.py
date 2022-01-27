#Class with methods for data loading, cleaning, feature extraction and anomaly detection from source
#Future methods to load streaming data and upload live forecast

import pandas as pd
from utils.crypto_utils import FeaturesExtractor


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

    def feature_extraction(self, cryptos_name, mode=1): # Feature extraction method
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

        mode: indicates if it extract data por classic or DL simulaiton
            - 0: classic
            - 1: DL
        '''

        fe = FeaturesExtractor()
        if mode == 1:
            i = 0
            for df in self.crypto_df:
                    df['High Low Difference'] = df['high'] - df['low']
                    df['Open Close Difference'] = df['open'] - df['close']
                    df['Result'] = df.apply(lambda row: fe.high_low(row), axis=1)                
        else: 
            i = 0
            for df in self.crypto_df:
                if self.cryptos_names[i] == cryptos_name:
                    df['High Low Difference'] = df['high'] - df['low']
                    df['Open Close Difference'] = df['open'] - df['close']
                    df['Sup 5'] = fe.get_support_resistance(df, 5, 'S')
                    df['Sup 50'] = fe.get_support_resistance(df, 50, 'S')
                    df['Sup 200'] = fe.get_support_resistance(df, 200, 'S')
                    df['Sup 500'] = fe.get_support_resistance(df, 500, 'S',)
                    df['Res 5'] = fe.get_support_resistance(df, 5, 'R')
                    df['Res 50'] = fe.get_support_resistance(df, 50, 'R')
                    df['Res 200'] = fe.get_support_resistance(df, 200, 'R')
                    df['Res 500'] = fe.get_support_resistance(df, 500, 'R')
                    df['SMA 50'] = fe.get_sma(df, 50)
                    df['SMA 200'] = fe.get_sma(df, 200)
                    df['upper_b_band'] = fe.get_bollinger_bands(df, 14, 2)[0]
                    df['lower_b_band'] = fe.get_bollinger_bands(df, 14, 2)[1]
                    df['EMA_50'] = fe.get_ema(df, 50)
                    df['EMA_200'] = fe.get_ema(df, 200)
                    df['MACD'] = fe.get_macd(df, 26, 12, 9)[0]
                    df['MACD_signal'] = fe.get_macd(df, 26, 12, 9)[1]
                    df['MACD_histo'] = fe.get_macd(df, 26, 12, 9)[2]
                    df['ADX'] = fe.get_adx(df, 14)[2]
                    df['Standard Deviation'] = fe.get_std(df, 14)
                    df['SO_K'] = fe.get_stoch(df, 14, 3)[0]
                    df['SO_D'] = fe.get_stoch(df, 14, 3)[1]
                    df['RSI'] = fe.get_rsi(df)
                    df['plus_di'] = fe.get_adx(df, 14)[0]
                    df['minus_di'] = fe.get_adx(df, 14)[1]
                    df['OBV'] = fe.get_obv(df)
                    df['OBV_diff'] = fe.get_obv_diff(df)
                    df['OBV_signal'] = fe.get_obv_signal(df['OBV_diff'], 20)
                    df['I_tenkan_sen'] = fe.get_ichimoku(df)[0]
                    df['I_kijun_sen'] = fe.get_ichimoku(df)[1]
                    df['I_senkou_span_a'] = fe.get_ichimoku(df)[2]
                    df['I_senkou_span_b'] = fe.get_ichimoku(df)[3]
                    df['I_chikou_span'] = fe.get_ichimoku(df)[4]
                    df['ATR'] = fe.get_atr(df, 14)
                    df['Result'] = df.apply(lambda row: fe.high_low(row), axis=1)
                i+=1

    def feature_selection(self, crypto_name, lvl): # Feature selection method
        '''
        Eliminates features not relevant or highly correlated to others
        '''
        i = 0
        for df in self.crypto_df:
            if self.cryptos_names[i] == crypto_name: # if crypto is already loaded
                #print('Dropping columns')
                if lvl == 0:
                    df.drop(columns=['SO_D', 'OBV_diff', 'plus_di', 'minus_di', 'MACD_histo'], inplace = True) # Elegir y value
                if lvl == 1:
                    self.crypto_df[self.cryptos_names.index(crypto_name)] = pd.concat([df.iloc[::, 1:16], df.iloc[::, -1]], axis=1) # Usar Result como y value
                if lvl == 2:
                    self.crypto_df[self.cryptos_names.index(crypto_name)] = pd.concat([df.iloc[::, 1], df.iloc[::,-4]], axis=1).iloc[:-1] # Usar shift como y value
            i+=1

    def lstm_processor(self, crypto_name): # prepares data for a lstm model
        '''
        Process data for LSTM model
        '''
        i = 0
        for df in self.crypto_df:
            if self.cryptos_names[i] == crypto_name: # if crypto is already loaded
                pass


    def detect_anomalies(self, crypto_name): # Anomaly detection method
        '''
        Build non supervised methods to detect outliers and anomalies and discard data
        '''