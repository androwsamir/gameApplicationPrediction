import pandas as pd
from sklearn.model_selection import train_test_split
import Preprocessing

def preprocess_input(filename):

    # Read data from CSV file to dataset
    dataset = pd.read_csv(filename)
    # split all column except last one for input in X
    X = dataset.iloc[:,:-1]
    # split last column for output in Y
    Y = dataset.iloc[:,-1]

    return X, Y


if __name__=='__main__':

    # Get input in X & output in Y
    X, Y = preprocess_input('games-regression-dataset.csv')

    # Splitting the X,Y into the Training set and Test set
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

    # Create Object of Preprocess Class
    preprocess = Preprocessing.Preprocess(x_train, x_test, y_train, y_test)
    
    #X['Age Rating'] = X['Age Rating'].str.replace('+' , '')
    #X['Age Rating'] = X['Age Rating'].astype(int)
    colums_lbl=('Developer','Primary Genre' , 'Languages' , 'Genres' ,'Age Rating')
    colums_hot=('Languages' , 'Genres')

    X =Preprocessing.pre_lbl_encoding(X , colums_lbl)
    #Preprocessing.pre_hot_encoding(X , colums_hot)
