class CryptoLSTM:
    '''
    LSTM solution for crypto market prediction

    Create the model, train and predict next values

    Different y values:
        - Close price
        - Difference open-hig
        - Result (0, 1)
    '''
    def __init__(self, df, y_value_selector = 0, periods_to_predict = 20):
        self.df = df
        self.y_value_selector = y_value_selector
        self.periods_to_predict = periods_to_predict
        self.model = 0

    def build(self):
        '''
        Build LSTM model
        '''

        
        pass

    def compile(self):
        '''
        Complie LSTM model
        '''
        pass

    def train(self):
        '''
        Train LSTM model
        '''
        pass

    def test(self):
        '''
        Run LSTM model
        '''
        pass

    def predict(self, model, df):
        '''
        Predict periods_to_predict based on current model
        '''
    
    def get_model(self):
        return self.model

    def set_dataset(self, df):
        self.df = df.copy()