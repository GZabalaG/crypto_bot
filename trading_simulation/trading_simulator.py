# Load data, select strategy, loop over data calling orders in crypto utils tradeops class

import pandas as pd

class TradingSimulator:

    def __init__(self, processor, crypto_name, strategy = [6], stop_loss_take_profit_strategy = 2, balance = 10000, loss_allowed=0.1, take_profit_mul = 3): # Constructor
        '''
        Simulator to calculate final result in balance after following a strategy
        processor: Crypto processor
        crypto_name: crypto(s) to apply strategy
        startegy: strategy(ies) to trigger buy
        stop_loss_take_profit_strategy: strategy to set stop loss and take profit
        balance: simulation initial balance
        loss_allowed: loss allowed in stop loss strategy in percentage points
        take_profit_mul: multiplier by loss_allowed to set take_profit strategy
        '''
        self.processor = processor
        self.crypto_name = crypto_name
        self.strategy = strategy
        self.stop_loss_take_profit_strategy = stop_loss_take_profit_strategy
        self.balance = balance
        self.df = self.init_crypto()
        column_names = ['close', 'total(€)', 'stop_loss', 'take_profit']
        self.orders = pd.DataFrame(columns = column_names)
        self.loss_allowed = loss_allowed
        self.take_profit_mul = take_profit_mul
        self.total_invest = 0
        self.orders_won = 0
        self.total_orders = 0
    
    def init_crypto(self):
        '''
        Variables initalization
        '''
        self.processor.load_data()
        self.processor.clean_data(self.crypto_name)
        self.processor.feature_extraction(self.crypto_name)
        df = self.processor.get_data(self.crypto_name)
        return df

    def simulate(self):
        '''
        For each row, check if any sell order needs to be made and then check if the strategy is appliable to the row.
        '''
        prev_row = self.df.iloc[0]
        for index, row in self.df.iterrows():
            self.check_orders(row['close'], row['date']) # Comprueba si se realiza algún sell de alguna order previa
            if(self.balance > 100):
                self.trigger_strategy(prev_row, row) # Comprueba si se realiza algún buy si se cumple alguna estrategia
            prev_row = row
        print('Total invested:', self.total_invest, '€')
        print('Balance:', self.balance, '€')
        print('Orders won:', self.orders_won)
        print('Orders lost:', self.total_orders - self.orders_won)

    def trigger_strategy(self, prev_row ,row):
        '''
        Returns true whenever the strategy is met for df (crypto) on index row
        '''
         # SMA or EMA - SO
        if 1 in self.strategy and row['SO_K'] > 20 and row['SMA 50'] >= row['SMA 200'] and prev_row['SMA 50'] < prev_row['SMA 200']:
            self.buy(row, strategy = 1)
        # BB - SO
        if 2 in self.strategy and row['SO_K'] > 20 and prev_row['close'] <= prev_row['lower_b_band'] and row['close'] > row['lower_b_band']:
            self.buy(row, strategy = 2)
        # MACD - RSI
        if 3 in self.strategy and row['RSI'] > 30 and prev_row['MACD'] < prev_row['MACD_signal'] and row['MACD'] >= row['MACD_signal']:
            self.buy(row, strategy = 3)
        # ADX - BB - RSI
        if 4 in self.strategy and row['RSI'] > 30 and row['ADX'] > 25 and row['ADX'] < 50 and prev_row['close'] <= prev_row['lower_b_band'] and row['close'] > row['lower_b_band']:
            self.buy(row, strategy = 4)
        # MACD
        if 5 in self.strategy and prev_row['MACD'] < prev_row['MACD_signal'] and row['MACD'] >= row['MACD_signal']:
            self.buy(row, strategy = 5)
        # BB
        if 5 in self.strategy and prev_row['close'] <= prev_row['lower_b_band'] and row['close'] > row['lower_b_band']:
            self.buy(row, strategy = 6)
        # OBV - RSI - BB
        if 6 in self.strategy and row['OBV_signal'] > 0.6 and row['RSI'] > 30 and prev_row['close'] <= prev_row['lower_b_band'] and row['close'] > row['lower_b_band']:
            self.buy(row, strategy = 7)
        # Ichimoku
        if 7 in self.strategy:
            self.buy(row, strategy = 8)

    def check_orders(self, close, date):
        '''
        Check any order to sell
        '''
        # if row stop loss order or take profit is met then apply balance changes and eliminate order
        for index, order in self.orders.iterrows():
            # update trailing stop loss strategy: subir stop loss si close price es menor de x%
            if self.stop_loss_take_profit_strategy == 2: 
                if order['stop_loss'] <= close-close*self.loss_allowed: order['stop_loss'] = close-close*self.loss_allowed

            if(close <= order['stop_loss'] or close >= order['take_profit']):
                self.sell(order, close, index, date) # Sell and drop order from orders dataframe

    def buy(self, row, strategy):
        '''
        Create order with stop_loss and take profit based on stop_loss_take_profit_strategy
        '''
        self.balance = self.balance - 102 #(comision)

        if self.stop_loss_take_profit_strategy == 1 and not pd.isnull(row['ATR']): # ATR
            stop_loss = row['close']-row['ATR']
            take_profit = row['close']+self.take_profit_mul*row['ATR']
            new_order = {'close_entry':row['close'], 'total(€)':100, 'stop_loss':stop_loss, 'take_profit':take_profit}
            self.orders = self.orders.append(new_order, ignore_index=True)
            self.total_invest += 100
            
            print('Buy on', row['date'],'||| Strat', strategy,' Close:', row['close'], '| Stop-loss:', stop_loss, '| Take-profit:', take_profit, ' | ATR:', row['ATR'], '|||')
        
        elif self.stop_loss_take_profit_strategy == 2: # Trailing
            stop_loss = row['close']-self.loss_allowed*row['close']
            take_profit = 99999999999999 # We can't never reach take profit, change in the future in the check orders
            new_order = {'close_entry':row['close'], 'total(€)':100, 'stop_loss':stop_loss, 'take_profit':take_profit}
            self.orders = self.orders.append(new_order, ignore_index=True)
            self.total_invest += 100
            
            print('Buy on', row['date'],'||| Strat', strategy,' Close:', row['close'], '| Stop-loss:', stop_loss, '| Take-profit:', take_profit, '|||')

        elif self.stop_loss_take_profit_strategy == 3: # % Loss - Profit
            stop_loss = row['close']-self.loss_allowed*row['close']
            take_profit = row['close']+self.take_profit_mul*self.loss_allowed*row['close']
            new_order = {'close_entry':row['close'], 'total(€)':100, 'stop_loss':stop_loss, 'take_profit':take_profit}
            self.orders = self.orders.append(new_order, ignore_index=True)
            self.total_invest += 100
            
            print('Buy on', row['date'],'||| Strat', strategy,' Close:', row['close'], '| Stop-loss:', stop_loss, '| Take-profit:', take_profit, '|||')

        else: # Sup - Res levels
            new_order = {'close_entry':row['close'], 'total(€)':100, 'stop_loss':row['Sup 50'], 'take_profit':row['Res 50']}
            self.orders = self.orders.append(new_order, ignore_index=True)
    
    def sell(self, order, close, index, date):
        '''
        Change balance based on order profit
        '''
        entry = order['close_entry']
        profit = close/entry
        if(profit > 1): self.orders_won += 1
        self.total_orders += 1
        self.balance += profit * order['total(€)']
        self.orders.drop(index, inplace=True) # Eliminate order from orders
        print('Sell order', index, 'on', date,' ||| Close:', close,'| Entry', entry,'| Profit:', profit,'| Balance:', self.balance, '|||')
