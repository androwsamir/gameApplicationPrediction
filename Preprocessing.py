import pandas as pd
import numpy as np

class Preprocess:

    # Constructor
    def __init__(self, df):
        self.df = df
        

    def Data_Cleaning(self):
        
        # Drop unuseful columns
        self.df.drop(['URL', 'ID', 'Name', 'Subtitle', 'Icon URL'], axis=1, inplace=True)
        
        # Fill the missing Languages value
        self.df['Languages'].fillna(value='EN', inplace=True)

        # Remove Duplicate Rows
        self.df.drop_duplicates(inplace=True)
        
        # Convert In-App Purchases from string to list of values
        # Example: "3.99, 1.99, 0.99"
        # To [3.99, 1.99, 0.99]
        self.df['In-app Purchases'] = self.df['In-app Purchases'].apply(lambda x: [float(value) for value in x.split(',')] if isinstance(x, str) else x)
        
        # Replace the list of values with the median
        self.df['In-app Purchases'] = self.df['In-app Purchases'].apply(lambda x: np.median(x) if x else None)

        # Replace Null values with the mean
        self.df['In-app Purchases'] = self.df['In-app Purchases'].fillna(self.df['In-app Purchases'].mean())
        
        # Convert Date Columns to datetime object
        self.df['Current Version Release Date'] = pd.to_datetime(self.df['Current Version Release Date'], format='%d/%m/%Y')
        self.df['Original Release Date'] = pd.to_datetime(self.df['Original Release Date'], format='%d/%m/%Y')
        
        # Add new feature from Date Columns
        self.df['Last Update Since Release Date'] = self.df.apply(lambda row: (row['Current Version Release Date'] - row['Original Release Date']).days, axis=1)
