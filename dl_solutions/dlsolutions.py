from keras.layers.core import Activation
from numpy.core.defchararray import startswith
from pandas._libs.tslibs import timestamps
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
    '''
    def __init__(self, df, norm_strat, strat, layers, neurons, batch_size, epochs, num_timestamps, num_features):
        '''
        df: dataset to use in model
        norm_strat: normalization strategy
        strat: model strategy
        layers: number of model layers
        neurons: number neurons per layers
        batch_size: batch size model
        epoch: training epochs
        num_timestamps: number of tiemstamps used in lstm model. If null features are pased as timesteps
        num_features: number of features per timesteps. Equal to 1 if timesteps are null (because is used as number of target features)
        '''
        self.df = df
        self.model = 0
        self.norm_strat = norm_strat
        self.strat = strat
        self.layers = layers
        self.neurons = neurons
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_timestamps = num_timestamps
        if self.num_timestamps is None:
            self.num_features = 1 #number of target features
        else:
            self.num_features = num_features

        # min max total
        self.min = df.max().max()
        self.max = df.min().min()
        # min max by column
        self.min_by_col = []
        self.max_by_col = []
        for col in df:
            min = df[col].min()
            max = df[col].max()
            self.min_by_col.append(min)
            self.max_by_col.append(max)
    
    def normalize(self, df_to_norm):
        '''
        We have to normalize all columns. Different strategies needed

        Strategies:
            0: normalize over max and min of whole dataset known
            1: normalize minmaxscaler
            2: minmax by columns considering whole dataset column
        '''

        #TODO create more norm strategies

        if self.norm_strat == 0:
            norm = (df_to_norm - self.min) / (self.max - self.min)
            return norm
        elif self.norm_strat == 1:
            self.sc = MinMaxScaler(feature_range = (0, 1))
            return self.sc.fit_transform(df_to_norm)
        elif self.norm_strat == 2:
            # Crear df con min max por columnas (extraer nombres columnas del df de entrada)
            # Por cada columna aplicar minmax con df de min max (iterar sobre df de min max)
            i = 0
            norm = []
            transpose = df_to_norm.transpose()
            for col in transpose:
                norm_col = (col - self.min_by_col[i])/(self.max_by_col[i] - self.min_by_col[i])
                norm.append(norm_col)
                i+=1
                
            norm = np.asarray(norm)
            return norm.transpose()

    def reverse_norm(self, df_to_norm):
        '''
        We have to return the normalize to all columns. Different strategies needed

        Strategies:
            0: normalize over max and min of whole dataset known
            1: normalize minmaxscaler
            2: minmax by columns considering whole dataset column
        '''
        
        #TODO create more norm strategies

        if self.norm_strat == 0:
            reverse_norm = df_to_norm * (self.max - self.min) + self.min
            return reverse_norm
        elif self.norm_strat == 1:
            return self.sc.inverse_transform(df_to_norm)
        elif self.norm_strat == 2:
            norm = []
            transpose = df_to_norm.transpose()
            i = 0
            for col in transpose:
                reverse_norm_col = col * (self.max_by_col[i] - self.min_by_col[i]) + self.min_by_col[i]
                norm.append(reverse_norm_col)
                i+=1
            norm = np.asarray(norm)
            return norm.transpose()

    def train_test_split(self):
        # split into train and test sets
        values = self.df.values
        train_frac = 1
        n_train_days = int(len(values)*train_frac)
        train = self.normalize(values[:n_train_days, :])
        test = self.normalize(values[n_train_days:, :])

        # split into input and outputs. We pass num_features as number of target features. Because timestamps
        # can be null, this value can be 1 if timestamps are null
        self.train_X, self.train_y = train[:, :-self.num_features], train[:, -self.num_features:]
        self.test_X, self.test_y = test[:, :-self.num_features], test[:, -self.num_features:]

        # reshape input to be 3D [samples, timesteps, features]
        # If timestamps is None we're defining timestamps as features so the whole row represents the features*timestamps
        # We need to reshape 'y' in case of use of timestamps
        if self.num_timestamps is None:
            self.train_X = self.train_X.reshape((self.train_X.shape[0], 1, self.train_X.shape[1]))
            self.test_X = self.test_X.reshape((self.test_X.shape[0], 1, self.test_X.shape[1]))
        else:
            self.train_X = self.train_X.reshape((self.train_X.shape[0], self.num_timestamps, self.num_features))
            self.test_X = self.test_X.reshape((self.test_X.shape[0], self.num_timestamps, self.num_features))
            self.train_y = self.train_y.reshape((self.train_y.shape[0], self.num_features))
            self.test_y = self.test_y.reshape((self.test_y.shape[0], self.num_features))

        print('Input shape:', self.train_X.shape, self.train_y.shape, self.test_X.shape, self.test_y.shape)
        
    def set_test(self, test_df):
        test = self.normalize(test_df.values)
        self.test_X, self.test_y = test[:, :-self.num_features], test[:, -self.num_features:]

        # reshape input to be 3D [samples, timesteps, features]
        # If timestamps is None we're defining timestamps as features so the whole row represents the features*timestamps
        # We need to reshape 'y' in case of use of timestamps
        if self.num_timestamps is None:
            self.test_X = self.test_X.reshape((self.test_X.shape[0], 1, self.test_X.shape[1]))
        else:
            self.test_X = self.test_X.reshape((self.test_X.shape[0], self.num_timestamps, self.num_features))
            self.test_y = self.test_y.reshape((self.test_y.shape[0], self.num_features))

    def build(self):
        '''
        Build LSTM model

        Strategies:
            0: binary
            1: mse

        '''

        #TODO parametrize layers and neurons


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
            self.model.add(Dense(units = self.num_features, activation='sigmoid'))
            # Compile
            self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        elif self.strat == 1:
            # Adding the output layer
            self.model.add(Dense(units = self.num_features, activation='relu'))
            # Compile
            self.model.compile(loss='mse', optimizer='adam')

    def train(self):
        '''
        Train LSTM model
        '''
        # Creates validation set
        n_val_days = int(len(self.train_X)*0.95)
        val_X = self.train_X[n_val_days:]
        val_y = self.train_y[n_val_days:]


        ###### TODO CREATE CALLBACKS
        callbacks = 0

        self.model.fit(self.train_X, self.train_y, epochs=self.epochs, batch_size=self.batch_size, validation_data=(val_X, val_y), verbose=2, shuffle=False, callbacks=callbacks)

    def predict(self):
        '''
        Return prediciton for test set
        '''
        # Make a prediction
        print('testx',self.test_X.shape)
        preds = self.model.predict(self.test_X)
        test_X = self.test_X.reshape((self.test_X.shape[0], self.test_X.shape[1] * self.test_X.shape[2]))
        print('preds', preds)

        # Invert scaling for forecast
        inv_preds = np.concatenate((test_X[:, :], preds), axis=1)
        inv_preds = self.reverse_norm(inv_preds)
        inv_preds = inv_preds[:,-self.num_features:] # antes -1
        print('preds rev norm',inv_preds)

        # Invert scaling for actual
        # test_y = self.test_y.reshape((len(self.test_y), 1))
        test_y = self.test_y.reshape((1, self.num_features))
        inv_y = np.concatenate((test_X[:, :], test_y), axis=1) #antes axis 1
        inv_y = self.reverse_norm(inv_y)
        inv_y = inv_y[:,-self.num_features:] # antes -1

        # calculate RMSE
        rmse = np.sqrt(mean_squared_error(inv_y, inv_preds))
        print('real', inv_y)
        print('Test RMSE: %.3f' % rmse)
        return inv_preds
    
    def get_model(self):
        return self.model

    def set_dataset(self, df):
        self.df = df.copy()