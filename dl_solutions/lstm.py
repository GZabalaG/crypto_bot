# LSTM solution for crypto market prediciton

class CryptoLSTM:

    def __init__(self, crypto_name, processor):
        processor.load_data()
        processor.clean_data(crypto_name)
        processor.feature_extraction(crypto_name)
        processor.feature_selection(crypto_name)
        self.df = processor.get_data(crypto_name)

    def build(self):
        pass

    def compile(self):
        pass

    def test(self):
        pass