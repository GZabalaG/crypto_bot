from keras.layers.core import Activation
from numpy.core.defchararray import startswith
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

class CryptoDLSolutions:
    '''
    LSTM solution for crypto market prediction

    Create the model, train and predict next values

    Different y values: it always will be the last column
        - Close price
        - Difference open-hig
        - Result (0, 1)
        - 
    '''
    def __init__(self, df, norm_strat, strat, layers, batch_size, epochs):
        '''
        y value is referred to the column that represents the model target
        '''
        self.df = df
        self.model = 0
        self.min = df.max().max()
        self.max = df.min().min()
        self.norm_strat = norm_strat
        self.strat = strat
        self.layers = layers
        self.batch_size = batch_size
        self.epochs = epochs
    
    def normalize(self, df_to_norm):
        '''
        We have to normalize all columns. Different strategies needed

        Strategies:
            0: normalize over max and min of whole dataset known
            1: normalize minmaxscaler
        '''

        if self.norm_strat == 0:
            norm = (df_to_norm - self.min) / (self.max - self.min)
            return norm
        elif self.norm_strat == 1:
            self.sc = MinMaxScaler(feature_range = (0, 1))
            return self.sc.fit_transform(df_to_norm)

    def reverse_norm(self, df_to_norm):
        '''
        We have to return the normalize to all columns. Different strategies needed

        Strategies:
            0: normalize over max and min of whole dataset known
            1: normalize minmaxscaler
        '''
        
        if self.norm_strat == 0:
            reverse_norm = df_to_norm * (self.max - self.min) + self.min
            return reverse_norm
        elif self.norm_strat == 1:
            return self.sc.inverse_transform(df_to_norm)

    def train_test_split(self):
        # split into train and test sets
        values = self.df.values
        n_train_days = int(len(values)*0.95)
        train = self.normalize(values[:n_train_days, :])
        test = self.normalize(values[n_train_days:, :])

        # split into input and outputs
        self.train_X, self.train_y = train[:, :-1], train[:, -1]
        self.test_X, self.test_y = test[:, :-1], test[:, -1]

        # reshape input to be 3D [samples, timesteps, features]
        self.train_X = self.train_X.reshape((self.train_X.shape[0], 1, self.train_X.shape[1]))
        self.test_X = self.test_X.reshape((self.test_X.shape[0], 1, self.test_X.shape[1]))
        print('Input shape:', self.train_X.shape, self.train_y.shape, self.test_X.shape, self.test_y.shape)

        
    def set_test(self, test_df):
        test = self.normalize(test_df.values)
        self.test_X, self.test_y = test[:, :-1], test[:, -1]
        self.test_X = self.test_X.reshape((self.test_X.shape[0], 1, self.test_X.shape[1]))

    def build(self):
        '''
        Build LSTM model

        Strategies:
            0: binary
            1: mse

        '''

        self.train_test_split()

        # design network
        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences = True, input_shape=(self.train_X.shape[1], self.train_X.shape[2])))

        # Adding a second LSTM layer and some Dropout regularisation
        self.model.add(LSTM(units = 50, return_sequences = True))
        self.model.add(Dropout(0.2))

        # Adding a third LSTM layer and some Dropout regularisation
        self.model.add(LSTM(units = 50, return_sequences = True))
        self.model.add(Dropout(0.2))

        # Adding a fourth LSTM layer and some Dropout regularisation
        self.model.add(LSTM(units = 50))
        self.model.add(Dropout(0.2))

        if self.strat == 0:
            # Adding the output layer
            self.model.add(Dense(units = 1, activation='sigmoid'))
            # Compile
            self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        elif self.strat == 1:
            # Adding the output layer
            self.model.add(Dense(units = 1, activation='relu'))
            # Compile
            self.model.compile(loss='mse', optimizer='adam')

    def train(self):
        '''
        Train LSTM model
        '''
        n_val_days = int(len(self.train_X)*0.95)
        val_X = self.train_X[n_val_days:]
        val_y = self.train_y[n_val_days:]


        ###### CREATE CALLBACKS
        callbacks = 0

        self.model.fit(self.train_X, self.train_y, epochs=self.epochs, batch_size=self.batch_size, validation_data=(val_X, val_y), verbose=2, shuffle=False, callbacks=callbacks)

    def predict(self):
        '''
        Return prediciton for test set
        '''
        ## habia tres puntos???
        # make a prediction
        print('testx',self.test_X.shape)
        print(self.test_X)
        preds = self.model.predict(self.test_X)
        test_X = self.test_X.reshape((self.test_X.shape[0], self.test_X.shape[2]))

        # invert scaling for forecast
        inv_preds = np.concatenate((test_X[:, :], preds), axis=1)
        inv_preds = self.reverse_norm(inv_preds)
        inv_preds = inv_preds[:,-1]

        # invert scaling for actual
        test_y = self.test_y.reshape((len(self.test_y), 1))
        inv_y = np.concatenate((test_X[:, :], test_y), axis=1)
        inv_y = self.reverse_norm(inv_y)
        inv_y = inv_y[:,-1]

        # calculate RMSE
        rmse = np.sqrt(mean_squared_error(inv_y, inv_preds))
        print('real', inv_y)
        print('Test RMSE: %.3f' % rmse)
        return inv_preds
    
    def get_model(self):
        return self.model

    def set_dataset(self, df):
        self.df = df.copy()