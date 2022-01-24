class CryptoLSTM:
    '''
    LSTM solution for crypto market prediction

    Create the model, train and predict next values

    Different y values:
        - Close price
        - Difference open-hig
        - Result (0, 1)
    '''
    def __init__(self, crypto_name, processor, y_value_selector = 0, periods_to_predict = 20):
        processor.load_data()
        processor.clean_data(crypto_name)
        processor.feature_extraction(crypto_name)
        processor.feature_selection(crypto_name)
        self.df = processor.get_data(crypto_name)
        self.y_value_selector = y_value_selector
        self.periods_to_predict = periods_to_predict

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

    def test(self):
        '''
        Run LSTM model
        '''
        pass

    def predict(self, model):
        '''
        Predict periods_to_predict based on current model
        '''