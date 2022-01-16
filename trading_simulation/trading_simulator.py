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
        column_names = ['close', 'total(€)', 'stop_loss', 'take_profit']
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
        if self.trigger_strategy(row):
            self.buy(row['close'])

    def trigger_strategy(self, row):
        # Logica que devuelve true si se cumple la condicion de la strategy actual
        if self.strategy == 1: # SMA - SO
            pass
        elif self.strategy == 2: # BB - SO
            pass
        elif self.strategy == 3: # MACD - RSI
            pass
        elif self.strategy == 4: # ADX - BB - RSI
            pass
        elif self.strategy == 5: # BB - MACD
            pass
        elif self.strategy == 6: # OBV - RSI - BB
            pass
        else: # Ichimoku
            pass

    def check_orders(self, close):
        # comprobar orders para ver si debemos retirar alguna
        # if row stop loss order or take profit is met then apply balance changes and eliminate order
        for index, order in self.orders.iterrows():
            # update trailing stop loss strategy: subir stop loss si close price es menor de x%
            if self.stop_loss_take_profit_strategy == 2: 
                if order['stop_loss'] <= close*ratio: order['stop_loss'] = close*ratio

            if(close <= order['stop_loss'] or close >= order['take_profit']):
                self.sell(order, close, index) # Sell and drop order from orders dataframe

    def buy(self, close):
        '''
        Create order with stop_loss and take profit based on stop_loss_take_profit_strategy
        '''
        self.balance = self.balance - 102 #(comision)

        if self.stop_loss_take_profit_strategy == 1: # ATR
            new_order = {'close_entry':close, 'total(€)':100, 'stop_loss':close, 'take_profit':close}
            self.orders = self.orders.append(new_order, ignore_index=True)
        
        elif self.stop_loss_take_profit_strategy == 2: # Trailing
            new_order = {'close_entry':close, 'total(€)':100, 'stop_loss':close, 'take_profit':close}
            self.orders = self.orders.append(new_order, ignore_index=True)
       
        elif self.stop_loss_take_profit_strategy == 3: # % Loss - Profit
            new_order = {'close_entry':close, 'total(€)':100, 'stop_loss':close, 'take_profit':close}
            self.orders = self.orders.append(new_order, ignore_index=True)

        else: # Sup - Res levels
            new_order = {'close_entry':close, 'total(€)':100, 'stop_loss':close, 'take_profit':close}
            self.orders = self.orders.append(new_order, ignore_index=True)

    
    def sell(self, order, close, index):
        '''
        Change balance based on order profit
        '''
        profit = close/order['close_entry']
        self.balance += profit * order['Total(€)']
        self.orders.drop(index) # Eliminate order from orders