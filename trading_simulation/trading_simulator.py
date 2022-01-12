# Load data, select strategy, loop over data calling orders in crypto utils tradeops class

from data_processor.data_processing import DataProcessor
import pandas as pd

class TradingSimulator:

    def __init__(self, crypto_name, strategy = 1, stop_loss_take_profit_strategy = 1, balance = 10000): # Constructor
        '''
        Simulator to calculate final result in balance after following a strategy
        '''
        self.crypto_name = crypto_name
        self.strategy = strategy
        self.stop_loss_take_profit_strategy = stop_loss_take_profit_strategy
        self.balance = balance
        self.df = self.init_crypto(crypto_name)
        column_names = ['order_id', 'close', 'total(€)', 'stop_loss', 'take_profit']
        self.orders = pd.DataFrame(columns = column_names)

    
    def init_crypto(self, crypto_name):
        processor = self.DataProcessor([crypto_name])
        processor.load_data()
        processor.clean_data(crypto_name)
        processor.feature_extraction(crypto_name)
        df = processor.get_data(crypto_name)
        return df

    def simulate(self):
        for index, row in self.df.iterrows():
            self.check_orders(row['close']) # Comprueba si se realiza algún sell de alguna order previa
            if(self.balance > 100):
                self.apply_strategy(row) # Comprueba si se realiza algún buy si se cumple alguna estrategia
        pass

    def apply_strategy(self, row):
        if self.trigger_strategy(self.strategy, row):
            self.buy(row)

    def trigger_strategy(self, strategy, row):
        # Logica que devuelve true si se cumple la condicion de la strategy actual
        return True

    def check_orders(self, close):
        # comprobar orders para ver si debemos retirar alguna
        # if row stop loss order or take profit is met then apply balance changes and eliminate order
        for order in self.orders.iterrows():
            # update trailing stop loss strategy: subir stop loss si close price es menor de x%
            if(close <= order['stop_loss'] or close >= order['take_profit']):
                self.sell(order)

    def buy(self, row):
        # Modify balance = balance - 102 (comision)
        # Create order with stop_loss and take profit based on stop_loss_take_profit_strategy
        pass
    
    def sell(self):
        # Modify balance = balance + % profit * 100
        # Eliminate order from orders
        pass