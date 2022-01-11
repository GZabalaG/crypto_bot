class TradingSimulator:

    from data_processor.data_processing import DataProcessor

    def __init__(self, crypto_name, strategy = 1, balance = 10000): # Constructor
        '''
        Simulator to calculate final result in balance after following a strategy
        '''
        self.crypto_name = crypto_name
        self.strategy = strategy
        self.balance = balance
        self.df = self.init_crypto(crypto_name)
    
    def init_crypto(self, crypto_name):
        processor = self.DataProcessor([crypto_name])
        processor.load_data()
        processor.clean_data(crypto_name)
        processor.feature_extraction(crypto_name)
        df = processor.get_data(crypto_name)
        return df

    def simulate(self):
        for index, row in self.df:
            self.check_orders(row)
            self.apply_strategy(row)
            # comporbar limits orders
            # aplicar estrategias
        pass

    def apply_strategy(self, row):
        # comprobar balance
        # trigger strategies
        if self.strategy == 1:
            pass
        elif self.strategy == 2:
            pass
        else:
            pass

    def check_orders(self, row):
        # comprobar orders para ver si debemos retirar alguna
        # if row stop loss order or take profit is met then apply balance changes and eliminate order
        pass

    def buy(self):
        pass
    
    def sell(self):
        pass

    def stop_order(self):
        pass
    
    def limit_order(self):
        pass

    def stop_limit_order(self):
        pass
# Load data, select strategy, loop over data calling orders in crypto utils tradeops class