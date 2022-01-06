# Utils methods for this project

class Features:

    def __init__(self): # Constructor
        pass

    def high_low(row):
        return 1 if row['Open Close Difference'] > 0 else 0

    def get_support(df, shift, op):
        '''
        Returns the support or resistance level of shift periods before based on maximum or minimum
        '''
        if op == 'S':
            for index, row in df.iterrows():
                shift = index + shift
                df.loc[index, 'Support'] = df.loc[index:shift, 'close'].max()
            return df['Support']

        elif op == 'R':
            for index, row in df.iterrows():
                shift = index + shift
                df.loc[index, 'Resistance'] = df.loc[index:shift, 'close'].min()
            return df['Resistance']

        else: 
            print('Error: not valid operation. R: resistance, S: support')