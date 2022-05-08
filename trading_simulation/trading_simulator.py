# Load data, select strategy, loop over data calling orders in crypto utils tradeops class

import pandas as pd
from dl_solutions.dlsolutions import CryptoDLSolutions
from data_processor.data_processing import DataProcessor

class TradingSimulator:
    '''
    Simulator for calssic solutions
    '''

    def __init__(self, processor, crypto_name, strategy = [6], stop_loss_take_profit_strategy = 2, balance = 10000, loss_allowed=0.1, take_profit_mul = 3, log = False): # Constructor
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
        self.initial_balance = balance
        self.df = self.init_crypto()
        column_names = ['close', 'total(€)', 'stop_loss', 'take_profit']
        self.orders = pd.DataFrame(columns = column_names)
        self.loss_allowed = loss_allowed
        self.take_profit_mul = take_profit_mul
        self.total_invest = 0
        self.orders_won = 0
        self.total_orders = 0
        self.log = log
    
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
        print('Profit:', (self.balance-self.initial_balance)/self.total_invest, '\n')

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
        if 6 in self.strategy and prev_row['close'] <= prev_row['lower_b_band'] and row['close'] > row['lower_b_band']:
            self.buy(row, strategy = 6)
        # OBV - RSI - BB
        if 7 in self.strategy and row['OBV_signal'] > 0.6 and row['RSI'] > 30 and prev_row['close'] <= prev_row['lower_b_band'] and row['close'] > row['lower_b_band']:
            self.buy(row, strategy = 7)
        # Ichimoku
        if 8 in self.strategy :
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
            
            if self.log: print('Buy on', row['date'],'||| Strat', strategy,' Close:', row['close'], '| Stop-loss:', stop_loss, '| Take-profit:', take_profit, ' | ATR:', row['ATR'], '|||')
        
        elif self.stop_loss_take_profit_strategy == 2: # Trailing
            stop_loss = row['close']-self.loss_allowed*row['close']
            take_profit = 99999999999999 # We can't never reach take profit, change in the future in the check orders
            new_order = {'close_entry':row['close'], 'total(€)':100, 'stop_loss':stop_loss, 'take_profit':take_profit}
            self.orders = self.orders.append(new_order, ignore_index=True)
            self.total_invest += 100
            
            if self.log: print('Buy on', row['date'],'||| Strat', strategy,' Close:', row['close'], '| Stop-loss:', stop_loss, '| Take-profit:', take_profit, '|||')

        elif self.stop_loss_take_profit_strategy == 3: # % Loss - Profit
            stop_loss = row['close']-self.loss_allowed*row['close']
            take_profit = row['close']+self.take_profit_mul*self.loss_allowed*row['close']
            new_order = {'close_entry':row['close'], 'total(€)':100, 'stop_loss':stop_loss, 'take_profit':take_profit}
            self.orders = self.orders.append(new_order, ignore_index=True)
            self.total_invest += 100
            
            if self.log: print('Buy on', row['date'],'||| Strat', strategy,' Close:', row['close'], '| Stop-loss:', stop_loss, '| Take-profit:', take_profit, '|||')

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
        if self.log: print('Sell order', index, 'on', date,' ||| Close:', close,'| Entry', entry,'| Profit:', profit,'| Balance:', self.balance, '|||')

class DLSimulator:
    '''
    Simulator for DL solutions

    Make predictions and decide stock operations guideed by the predictions
    '''

    def __init__(self, crypto, prev_periods, pred_periods, columns, target,
    norm_strat, model_sel, layers, neurons, batch_size, epochs, 
    activation, loss, metrics, optimizer, initial_learning_rate, callbacks):
        '''
        periods_to_re...: periods to retrain model
        prev_periods: periodos usados como X
        pred_periods: periods shifted / periods to predict
        columns: columns to be selected from df processed to be used in model. Last column is target columns
        
        self.periods_to_retraining = periods_to_retraining
        self.prev_periods = prev_periods
        self.pred_periods = pred_periods
        self.model_selector = model_selector
        processor.load_data()
        processor.clean_data(crypto_name)
        processor.feature_extraction(crypto_name)
        #columns = ['close','Volume USDT' ,'Result']
        #columns = ['close']
        columns = columns
        self.df = processor.feature_selection(crypto_name, columns) # de aqui ya sale un df con X columns y target column en la ultima column
        self.df = processor.lstm_processing(self.df, target, prev_periods, pred_periods) # columnas con shift

        self.model = 0

        order_column_names = ['close', 'total(€)', 'stop_loss', 'take_profit']
        self.orders = pd.DataFrame(columns = order_column_names)
        '''
        
        self.crypto = crypto
        self.prev_periods = prev_periods
        self.pred_periods = pred_periods
        self.columns = columns
        self.target = target
        self.norm_strat = norm_strat
        self.model_sel = model_sel
        self.layers = layers
        self.neurons = neurons
        self.batch_size = batch_size
        self.epochs = epochs
        self.activation = activation
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.initial_learning_rate = initial_learning_rate
        self.callbacks = callbacks

        self.columns = columns
        self.target = target
        self.prev_periods = prev_periods
        self.pred_periods = pred_periods
        self.num_features = len(columns)

        if target != None: 
            self.num_timestamps = None
        else:
            self.num_timestamps = prev_periods

        self.processor = DataProcessor([self.crypto])
        self.tranforms_df()

        self.lstm = CryptoDLSolutions(self.df, norm_strat, model_sel, layers, neurons, batch_size, epochs, self.num_timestamps, 
        self.num_features, activation, loss, metrics, optimizer, initial_learning_rate, callbacks)

        order_column_names = ['close', 'value', 'stop_loss', 'take_profit']
        self.orders = pd.DataFrame(columns = order_column_names)

        p_fake_real_df = 0.96
        self.start_index = int(len(self.df) * p_fake_real_df)

        self.closed_orders = []
        self.open_order = 0
        self.order_goal = 0
        self.buy_sell_mode = 'buy'
        self.risk = 0.3 # Max loss allowance
        self.p_buy = 0.2 # How much higher the value has to be than todays price to buy. Always greater than goal_error
        self.spread = 0.02
        self.predicted_values = []

    def tranforms_df(self):
        self.processor.load_data()
        self.processor.clean_data(self.crypto)
        self.processor.feature_extraction(self.crypto)
        self.df = self.processor.feature_selection(self.crypto, self.columns)
        self.df = self.processor.lstm_processing(self.df, self.target, self.prev_periods, self.pred_periods)
        self.df.dropna(inplace=True)

    def check_orders(self, todays):
        '''
        Check made orders to sell if needed
        '''
        print('CHECKING ORDERS...')
        # Loss allowance
        if (self.open_order-todays)/self.open_order > self.risk:
            self.closed_orders.append(todays/self.open_order)
            self.buy_sell_mode = 'buy'
            print('### LOSS ALLOWANCE SELL ###')
            print('TODAY VALUE:', todays)
            print('ORDER VALUE:', self.open_order)
            print('PREDICTED VALUE:', self.order_goal)
        # Predicted goal
        elif (todays >= self.order_goal):
            self.closed_orders.append(self.open_order/todays)
            self.buy_sell_mode = 'buy'
            print('### PREDICTED GOAL SELL ###')
            print('TODAY VALUE:', todays)
            print('ORDER VALUE:', self.open_order)
            print('PREDICTED VALUE:', self.order_goal)

    def train_model(self, df):
        '''
        Train model with features applied in constructor
        df: dataset from init to todays value
        Before training tranformation of df is applied
        '''

        # Set df to model object
        self.lstm.set_dataset(df)

        # Train
        self.lstm.build()
        self.lstm.train()

    def get_history(self):
        return self.lstm.get_history()
    
    def get_model(self):
        return self.lstm.get_model()

    def predict(self, row):
        self.lstm.set_test(row)
        return self.lstm.predict()

    def apply_orders(self, todays, predicted):
        '''
        Apply orders based on predicted value and strategy
        '''
        # Si el valor predicho esta por encima en un p_buy % del valor actual, compramos

        if self.buy_sell_mode == 'buy':
            if (predicted-todays)/todays > self.p_buy:
                self.open_order = todays
                self.order_goal = predicted
                self.buy_sell_mode = 'sell'
                print('### BUY OPERATION ###')
                print('VALUE TODAY:', todays)
                print('VALUE PREDICTED IN 5 DAYS:', self.order_goal)
        else:
            # Predicted > previously predicted
            if (predicted-self.order_goal)/predicted >= self.p_buy:
                self.order_goal = predicted
                print('### PREDICTED VALUE UPDATE ###')
                print('VALUE TODAY:', todays)
                print('VALUE PREDICTED IN 5 DAYS:', self.order_goal)
                
            # Si la prediccion ha bajado pero ya estamos ganando dinero cerramos
            elif (todays-self.open_order)/todays > self.spread:
                    self.closed_orders.append(todays/self.open_order)
                    self.buy_sell_mode = 'buy'
                    print('### PREDICTED VALUE - NEW PREDICTED LOWER - WIN POSITION ###')
                    print('VALUE TODAY:', todays)
                    print('VALUE ORDER:', self.open_order)

    def simulate(self):
        i = 0
        print('Simulation starting...')
        for index, row in self.df.iterrows():
            if index > self.start_index:
                print('INDEX:', index)
                print('CLOSE:', row['close_0'])
                print('MODE:', self.buy_sell_mode)
                
                if self.buy_sell_mode == 'sell':
                    #Check previous orders
                    self.check_orders(row['close_0'])
                
                df_train = self.df.iloc[:index]
                df_test = self.df.iloc[index:index+1]

                #Train model
                print('TRAINING MODEL:')
                #print('TRAIN LAST ROWS:', self.df.iloc[index-5:index])
                self.train_model(self.df.iloc[:index])
                
                #Predict next value
                print('PREDICTING VALUE FOR', self.df.iloc[index:index+1])
                pred = self.predict(self.df.iloc[index:index+1])
                self.predicted_values.append(pred)

                #Apply new orders 
                self.apply_orders(row['close_0'], pred)

                print('||||----|||| Current Orders:', self.closed_orders, '\n\n\n')
                if i==30: break
                i+=1

        return self.start_index, self.predicted_values, self.closed_orders

    def get_df(self):
        return self.df