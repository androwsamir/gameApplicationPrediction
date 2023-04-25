import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
import Preprocessing
import Regression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import saveLoadData
import test

def preprocess_input(filename):

    # Read data from CSV file to dataset
    dataset = pd.read_csv(filename)
    # Split all column except last one for input in X
    X = dataset.iloc[:,:-1]
    # Split last column for output in Y
    Y = dataset.iloc[:,-1]

    return X, Y

def featureSelection(data, input):

    selector = SelectKBest(f_regression, k=5)
    inputData = selector.fit_transform(input, data['Average User Rating'])
    # Get the names of the selected features
    mask = selector.get_support()  # Get a boolean mask of the selected features
    top_feature = input.columns[mask]  # Index the feature names using the mask

    # Print the selected feature names
    saveLoad.saveModel(top_feature, 'topFeatures')
    print(top_feature)
    return inputData

def calculateMSE(prediction, actualValue):
    # Display MSE value for feature
    print('Mean Square Error: ' + str(mean_squared_error(actualValue, prediction)))

def calculateR2Score(prediction, actualValue):
    # Display R2 Score
    print(f'R2 Score : {r2_score(actualValue, prediction)}')

if __name__=='__main__':

    # Create object to load/save data/models in/from file
    saveLoad = saveLoadData.SaveLoadData()

    # Get input in X & output in Y
    X, Y = preprocess_input('games-regression-dataset.csv')

    # Splitting the X,Y into the Training set(80%) and Test set(20%)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

    # Create object of Preprocess Class
    preprocess = Preprocessing.Preprocess(x_train, y_train)

    # Clean Training Data
    x_train, y_train = preprocess.trainDataCleaning()

    # Concatenate x_train and y_train in data
    data = pd.concat([pd.DataFrame(x_train), pd.DataFrame(y_train)], axis=1)

    # Apply feature selection on x_train
    x_train = featureSelection(data, x_train)

    # Create object of Regression class and set constructor to x_train, y_train
    regression = Regression.regression(x_train, y_train)

    # Train models on x_train, y_train and predict x_train
    y_poly, y_lasso, y_decision, rfPrediction, svrPrediction, enetPrediction = regression.train()

    print("=====================================Train=============================================")
    # print MSE
    print("MSE Polynomial ==> ")
    calculateMSE(np.asarray(y_poly), np.asarray(y_train))
    calculateR2Score(y_poly, y_train)

    print("MSE lasso Regression ==> ")
    calculateMSE(np.asarray(y_lasso), np.asarray(y_train))
    calculateR2Score(y_lasso, y_train)

    print("MSE decision Regression ==> ")
    calculateMSE(np.asarray(y_decision), np.asarray(y_train))
    calculateR2Score(y_decision, y_train)

    print("MSE RandomForest Regression ==> ")
    calculateMSE(np.asarray(rfPrediction), np.asarray(y_train))
    calculateR2Score(rfPrediction, y_train)

    print("MSE Elastic Regression ==> ")
    calculateMSE(np.asarray(enetPrediction), np.asarray(y_train))
    calculateR2Score(enetPrediction, y_train)

    print("MSE SVR Regression ==> ")
    calculateMSE(np.asarray(svrPrediction), np.asarray(y_train))
    calculateR2Score(svrPrediction, y_train)

    testPredict = test.TestPredict(x_test)
    xTest = testPredict.preprocess()
    topFeature = saveLoad.loadModel('topFeatures')
    xTest = xTest[topFeature]
    polyPredict, lassoPredict, decisionTreePredict, rfPrediction, svrPrediction, enetPrediction = testPredict.predict(xTest)

    print("=====================================Test=============================================")
    # print MSE
    print("MSE Polynomial ==> ")
    calculateMSE(np.asarray(polyPredict), np.asarray(y_test))
    calculateR2Score(polyPredict, y_test)

    print("MSE lasso Regression ==> ")
    calculateMSE(np.asarray(lassoPredict), np.asarray(y_test))
    calculateR2Score(lassoPredict, y_test)

    print("MSE decision Regression ==> ")
    calculateMSE(np.asarray(decisionTreePredict), np.asarray(y_test))
    calculateR2Score(decisionTreePredict, y_test)

    print("MSE RandomForest Regression ==> ")
    calculateMSE(np.asarray(rfPrediction), np.asarray(y_test))
    calculateR2Score(rfPrediction, y_test)

    print("MSE Elastic Regression ==> ")
    calculateMSE(np.asarray(enetPrediction), np.asarray(y_test))
    calculateR2Score(enetPrediction, y_test)

    print("MSE SVR Regression ==> ")
    calculateMSE(np.asarray(svrPrediction), np.asarray(y_test))
    calculateR2Score(svrPrediction, y_test)