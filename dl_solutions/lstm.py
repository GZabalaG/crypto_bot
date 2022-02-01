from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import pandas as pd
import numpy as np

class CryptoLSTM:
    '''
    LSTM solution for crypto market prediction

    Create the model, train and predict next values

    Different y values: it always will be the last column
        - Close price
        - Difference open-hig
        - Result (0, 1)
        - 
    '''
    def __init__(self, df, periods_to_predict = 20):
        '''
        y value is referred to the column that represents the model target
        '''
        self.df = df
        self.periods_to_predict = periods_to_predict
        self.model = 0
    
    def normalize(self, strat):
        '''
        We have to normalize all columns. Different strategies needed
        '''
        pass

    def train_test_split(self):
        # split into train and test sets
        values = self.df.values
        n_train_days = 850
        train = values[:n_train_days, :]
        test = values[n_train_days:, :]
        # split into input and outputs
        self.train_X, self.train_y = train[:, :-1], train[:, -1]
        self.test_X, self.test_y = test[:, :-1], test[:, -1]
        # reshape input to be 3D [samples, timesteps, features]
        self.train_X = self.train_X.reshape((self.train_X.shape[0], 1, self.train_X.shape[1]))
        self.test_X = self.test_X.reshape((self.test_X.shape[0], 1, self.test_X.shape[1]))
        print('Input shape:', self.train_X.shape, self.train_y.shape, self.test_X.shape, self.test_y.shape)

    def build(self):
        '''
        Build LSTM model
        '''
        
        self.normalize(0)
        self.train_test_split()
        # design network
        self.model = Sequential()
        self.model.add(LSTM(50, input_shape=(self.train_X.shape[1], self.train_X.shape[2])))
        self.model.add(Dense(1))

    def compile(self):
        '''
        Complie LSTM model
        '''
        self.model.compile(loss='mae', optimizer='adam')

    def train(self):
        '''
        Train LSTM model
        '''
        self.model = self.model.fit(self.train_X, self.train_y, epochs=50, batch_size=72, validation_data=(self.test_X, self.test_y), verbose=2, shuffle=False)


    def test(self):
        '''
        Run LSTM model
        '''
        pass

    def predict(self, df):
        '''
        Predict periods_to_predict based on model
        '''
        ...
        # make a prediction
        yhat = self.model.predict(self.test_X)
        test_X = self.test_X.reshape((self.test_X.shape[0], self.test_X.shape[2]))
        # invert scaling for forecast
        inv_yhat = pd.concatenate((yhat, test_X[:, 1:]), axis=1)
        #inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:,0]
        # invert scaling for actual
        test_y = self.test_y.reshape((len(self.test_y), 1))
        inv_y = pd.concatenate((test_y, test_X[:, 1:]), axis=1)
        #inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:,0]
        # calculate RMSE
        rmse = pd.sqrt(pd.mean_squared_error(inv_y, inv_yhat))
        print('Test RMSE: %.3f' % rmse)
    
    def get_model(self):
        return self.model

    def set_dataset(self, df):
        self.df = df.copy()