import Preprocessing
import Regression
import pandas as pd

class TestPredict:

    def __init__(self, xTest=None):
        self.X = xTest

    def readData(self, filename):
        # Read csv file
        dataset = pd.read_csv(filename)
        # set dataset to X
        self.X = dataset

    def preprocess(self):
        # Create object of Preprocess
        preproces = Preprocessing.Preprocess(self.X)
        # Clean data
        self.X = preproces.testDataCleaning()

        return self.X

    def predict(self, xTest):
        # Create object of regression and set constructor to xTest
        regression = Regression.regression(xTest)
        # Get prediction of xtest
        polyPredict, lassoPredict, decisionTreePredict, rfPrediction, svrPrediction, enetPrediction = regression.test()

        return polyPredict, lassoPredict, decisionTreePredict, rfPrediction, svrPrediction, enetPrediction