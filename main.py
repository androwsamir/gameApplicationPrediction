import pandas as pd
from sklearn.model_selection import train_test_split
import Preprocessing


if __name__== '__main__':

    dataset = pd.read_csv('games-regression-dataset.csv')

    preprocess = Preprocessing.Preprocess(dataset)
    
    preprocess.Data_Cleaning()
    
    X = dataset.drop('Average User Rating', axis=1)
    
    Y = dataset['Average User Rating']
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
