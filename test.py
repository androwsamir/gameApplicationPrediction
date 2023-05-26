import Preprocessing
import Regression
import pandas as pd
import Classification

class TestPredict:

    def __init__(self, xTest=None):
        self.X = xTest

    def readData(self, filename):
        # Read csv file
        dataset = pd.read_csv(filename)
        # set dataset to X
        self.X = dataset.iloc[:,:]

    def preprocess(self):
        # Create object of Preprocess
        preproces = Preprocessing.Preprocess(self.X)
        # Clean data
        self.X = preproces.testDataCleaning()

        return self.X

    def predictRegression(self, xTest):
        # Create object of regression and set constructor to xTest
        regression = Regression.regression(xTest)
        # Get prediction of xtest
        polyPredict, decisionTreePredict, rfPrediction, enetPrediction = regression.test()

        return polyPredict, decisionTreePredict, rfPrediction, enetPrediction

    def predictClassification(self, xTest):
        # Create object of classification and set constructor to xTest
        classification = Classification.classification(xTest)

        return classification.testData()