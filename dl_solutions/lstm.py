from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

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
        self.train_X, self.train_y, self.test_X, self.test_y = 0

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
    
    def get_model(self):
        return self.model

    def set_dataset(self, df):
        self.df = df.copy()