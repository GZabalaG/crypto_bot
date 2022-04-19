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
    #### 1 STEP ####
    crypto = 'ETH'
    prev_periods = 5
    pred_periods = 10
    model_selector = 'lstm'
    columns = ['RSI', 'close']
    num_features = len(columns)
    target = None
    #target = 'close'


    #### STEP 2 ####
    norm_strat = 2
    model_sel = 0
    layers = 3
    neurons = [50, 50, 50, 50]
    batch_size = 64
    epochs = 150
    activations = ['relu', 'sigmoid']
    losses = ['mse', 'binary_crossentropy']
    activation = 'relu'
    loss = 'mse'
    metrics = ['mse']
    optimizer = 'adam'
    initial_learning_rate = 0.01

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
        self.columns = columns
        self.target = target
        self.prev_periods = prev_periods
        self.pred_periods = pred_periods
        num_features = len(columns)
        if target != None: 
            num_timestamps = None
        else:
            num_timestamps = prev_periods

        self.processor = DataProcessor([self.crypto])
        self.tranforms_df()

        self.lstm = CryptoDLSolutions(self.df, norm_strat, model_sel, layers, neurons, batch_size, epochs, num_timestamps, 
        num_features, activation, loss, metrics, optimizer, initial_learning_rate, callbacks)

        order_column_names = ['close', 'value', 'stop_loss', 'take_profit']
        self.orders = pd.DataFrame(columns = order_column_names)

        p_fake_real_df = 0.75
        self.start_index = int(len(self.df) * p_fake_real_df)

    def tranforms_df(self):
        self.processor.load_data()
        self.processor.clean_data(self.crypto)
        self.processor.feature_extraction(self.crypto)
        self.df = self.processor.feature_selection(self.crypto, self.columns)
        self.df = self.processor.lstm_processing(self.df, self.target, self.prev_periods, self.pred_periods)
        self.df.dropna(inplace=True)

    def check_orders(self):
        '''
        Check made orders to sell if needed
        '''
        #TODO

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

    def apply_orders(self, predicted, strat):
        '''
        Apply orders based on predicted value and strategy
        '''
        #TODO

    def simulate(self):
        #TODO start iteration in last row of real df
        for index, row in self.df.iterrows():
            if index > self.start_index:
           
                #Check previous orders
                self.check_orders()

                #Train model
                self.train_model(self.df.iloc[:index])
                
                #Predict next value
                pred = self.predict(row)

                #Apply new orders 
                self.apply_orders(pred, self.strat)

                break


    def get_df(self):
        return self.df

    def re_train_model(self, index):
        '''
        Train new model based on df
        '''
        # Get only data processed by simulation
        df_to_train = self.df[:index].copy() 
        
        if(self.model_selector == 'lstm'):
            self.model = CryptoDLSolutions(df_to_train)
        elif(self.model_selector == 'tcn'):
            pass
        
        self.model.build()
        self.model.compile()
        self.model.train()
        
    def make_predictions(self, index, days_to_predict):
        '''
        Fill old predicted data with new data
        Predict next N steps ahead

        We take in account pred_periods so predict() return the today period + pred_periods prediction
        '''
        # First we predict the new row. Valorar entrenar lstm diferentes por cada medida y hacer caluclos en base a las predicciones de todas las medidas predichas

        # Second we decide what to do what the predicted data

        n_periods = 20
        #df_to_predict = self.df.iloc[index, :-1].copy() # probar a pasar solo fila a predecir o multiples filas anteriores 
        df_to_predict = self.df.iloc[index-n_periods:, :-1].copy()
        preds = self.model.predict(df_to_predict) # Devuelve y values
        pred_final = preds[-1]
        return self.model.predict(df_to_predict)

    def simulate2(self):
        '''
        WE'LL NEED TO TRAIN EACH NEW VALUE. UPDATE WITH NEW REAL VALUES EACH ROW

        For each M periods
            re train model
            re fill predicted data with real one
        For each N periods
            Make predictions
            Validate old operations
            Make new stock operations
        '''
        for index, row in self.df.iterrows():
            if index % self.periods_to_retraining == 0 and index >= self.periods_to_retraining:
                print('Training model at:', index)
                self.re_train_model(index)
            if index % self.pred_periods == 0 and index > self.periods_to_retraining:
                print('Making predictions at:', index)
                predictions = self.make_predictions(index, 20)
                print('Validating and making orders at:', index)
                self.validate_and_make_orders(predictions)
    
    def validate_and_make_orders(self, predictions):
        '''
        Algorithm to decide orders based on new predictions
        '''
        pass
