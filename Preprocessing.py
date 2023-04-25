
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

class Preprocess:

    # Constructor
    def __init__(self, x_train, x_test, y_train, y_test):
        self.xTrain = x_train
        self.xTest = x_test
        self.yTrain = y_train
        self.yTest = y_test
        
        
    def pre_lbl_encoding( data , colums):
        for x in colums:
            le = LabelEncoder()
            le.fit(list(data[x].values))
            data[x] = le.transform(list(data[x].values))
        return data

    def pre_hot_encoding( data , colums):
        for x in colums:
            enc = OneHotEncoder()
            enc.fit(list(list(data[x].values)))
            data[x] = enc.transform(list(list(data[x].values)))
        return data
        
    
