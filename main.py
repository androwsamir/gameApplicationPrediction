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

def featureSelectionNumerical(data, input):
    # Compute the correlation matrix
    corr_matrix = data.corr()
    # Select top features based on correlation with output variable
    top_featureNumerical = corr_matrix.index[abs(corr_matrix['Average User Rating']) > 0.05]
    top_featureNumerical = top_featureNumerical.delete(-1)
    saveLoad.saveModel(top_featureNumerical, 'topFeatureNumerical')
    inputData = input[top_featureNumerical]

    return inputData

def featureSelectionCategorical(data, input):

    selector = SelectKBest(f_regression, k=5)
    inputData = selector.fit_transform(input, data['Average User Rating'])
    # Get the names of the selected features
    mask = selector.get_support()  # Get a boolean mask of the selected features
    top_featureCategorical = input.columns[mask]  # Index the feature names using the mask

    inputData = pd.DataFrame(inputData, columns=top_featureCategorical)

    # Print the selected feature names
    saveLoad.saveModel(top_featureCategorical, 'topFeaturesCategorical')
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

    x_train =x_train.copy()
    xNumerical = x_train.iloc[:, 0:1]
    xNumerical = xNumerical.copy()
    xNumerical['Size'] = x_train['Size']

    xCategorical = x_train.iloc[:, 5:]
    xCategorical['Developer'] = x_train['Developer']
    xCategorical['Age Rating'] = x_train['Age Rating']

    # Concatenate x_train and y_train in data
    categoricalData = pd.concat([pd.DataFrame(xCategorical.reset_index(drop=True)), pd.DataFrame(y_train.reset_index(drop=True))], axis=1)
    numericalData = pd.concat([pd.DataFrame(xNumerical.reset_index(drop=True)), pd.DataFrame(y_train.reset_index(drop=True))], axis=1)

    xNumerical = featureSelectionNumerical(numericalData, xNumerical)

    xCategorical = featureSelectionCategorical(categoricalData, xCategorical)


    x_train = pd.concat([pd.DataFrame(xCategorical.reset_index(drop=True)), pd.DataFrame(xNumerical.reset_index(drop=True))], axis=1)

    # Create object of Regression class and set constructor to x_train, y_train
    regression = Regression.regression(x_train, y_train)

    # Train models on x_train, y_train and predict x_train
    y_poly, y_decision, rfPrediction, enetPrediction = regression.train()

    print("=====================================Train=============================================")
    # print MSE
    print("Polynomial ==> ")
    calculateMSE(np.asarray(y_poly), np.asarray(y_train))
    calculateR2Score(y_poly, y_train)

    print("decision Regression ==> ")
    calculateMSE(np.asarray(y_decision), np.asarray(y_train))
    calculateR2Score(y_decision, y_train)

    print("RandomForest Regression ==> ")
    calculateMSE(np.asarray(rfPrediction), np.asarray(y_train))
    calculateR2Score(rfPrediction, y_train)

    print("Elastic Regression ==> ")
    calculateMSE(np.asarray(enetPrediction), np.asarray(y_train))
    calculateR2Score(enetPrediction, y_train)

    testPredict = test.TestPredict(x_test)
    xTest = testPredict.preprocess()
    topFeatureNumerical = saveLoad.loadModel('topFeatureNumerical')
    topFeaturesCategorical = saveLoad.loadModel('topFeaturesCategorical')

    xTest = pd.concat([pd.DataFrame(xTest[topFeatureNumerical]), pd.DataFrame(xTest[topFeaturesCategorical])], axis=1)

    polyPredict, decisionTreePredict, rfPrediction, enetPrediction = testPredict.predict(xTest)

    print("=====================================Test=============================================")
    # print MSE
    print("Polynomial ==> ")
    calculateMSE(np.asarray(polyPredict), np.asarray(y_test))
    calculateR2Score(polyPredict, y_test)

    print("decision Regression ==> ")
    calculateMSE(np.asarray(decisionTreePredict), np.asarray(y_test))
    calculateR2Score(decisionTreePredict, y_test)

    print("RandomForest Regression ==> ")
    calculateMSE(np.asarray(rfPrediction), np.asarray(y_test))
    calculateR2Score(rfPrediction, y_test)

    print("Elastic Regression ==> ")
    calculateMSE(np.asarray(enetPrediction), np.asarray(y_test))
    calculateR2Score(enetPrediction, y_test)
