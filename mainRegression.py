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
import csv
import matplotlib.pyplot as plt

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
    top_featureNumerical = corr_matrix.index[abs(corr_matrix['Average User Rating']) > 0.5]
    top_featureNumerical = top_featureNumerical.delete(-1)
    # print(top_featureNumerical)
    saveLoad.saveModel(top_featureNumerical, 'topFeatureNumericalRegression')
    inputData = input[top_featureNumerical]

    return inputData

def featureSelectionCategorical(data, input):

    selector = SelectKBest(f_regression, k=2)
    inputData = selector.fit_transform(input, data['Average User Rating'])
    # Get the names of the selected features
    mask = selector.get_support()  # Get a boolean mask of the selected features
    top_featureCategorical = input.columns[mask]  # Index the feature names using the mask

    inputData = pd.DataFrame(inputData, columns=top_featureCategorical)

    # Print the selected feature names
    saveLoad.saveModel(top_featureCategorical, 'topFeaturesCategoricalRegression')
    return inputData

def calculateMSE(prediction, actualValue):
    # Display MSE value for feature
    print('Mean Square Error: ' + str(mean_squared_error(actualValue, prediction)))

def calculateR2Score(prediction, actualValue):
    # Display R2 Score
    print(f'R2 Score : {r2_score(actualValue, prediction)}')

def plot_model_fit(model_name, y_pred, y_actual):
    plt.scatter(y_actual, y_pred, color='blue', label='Actual vs. Predicted')
    plt.scatter(y_actual, y_pred, color='red', label='Actual vs. Predicted (Different Color)')
    plt.plot([min(y_actual), max(y_actual)], [min(y_actual), max(y_actual)], color='red', label='Perfect Fit Line')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Model Fit - {model_name}')
    plt.legend()
    plt.show()

def plot_regression_line(model_name, y_pred, y_actual):
    plt.scatter(y_actual, y_pred, color='blue', label='Actual vs. Predicted')
    plt.plot([min(y_actual), max(y_actual)], [min(y_actual), max(y_actual)], color='red', label='Regression Line')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Regression Line - {model_name}')
    plt.legend()
    plt.show()

if __name__=='__main__':
    x = input("Enter 1 if you want run train , 2 for test : ")

    # Create object to load/save data/models in/from file
    saveLoad = saveLoadData.SaveLoadData()

    if x == '1':
        # Get input in X & output in Y
        X, Y = preprocess_input('games-regression-dataset.csv')

        # Splitting the X,Y into the Training set(80%) and Test set(20%)
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

        # Create object of Preprocess Class
        preprocess = Preprocessing.Preprocess(x_train, y_train)

        # Clean Training Data
        x_train, y_train = preprocess.trainDataCleaning()
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.expand_frame_repr', False)

        doc2vec_cols = x_train.filter(regex='^doc2vec_')
        x_train = x_train.copy()
        xNumerical = x_train.iloc[:, 0:5]
        tmpList = ['Developer', 'Age Rating']
        for col in tmpList:
            xNumerical.drop(col, axis=1, inplace=True)
        xNumerical = pd.concat(
            [pd.DataFrame(xNumerical.reset_index(drop=True)), pd.DataFrame(doc2vec_cols.reset_index(drop=True))],
            axis=1)

        doc2vecNames = doc2vec_cols.columns
        xCategorical = x_train.iloc[:, 5:]
        xCategorical['Developer'] = x_train['Developer']
        xCategorical['Age Rating'] = x_train['Age Rating']

        xCategorical = xCategorical.drop(doc2vecNames, axis=1)

        # Concatenate x_train and y_train in data
        categoricalData = pd.concat(
            [pd.DataFrame(xCategorical.reset_index(drop=True)), pd.DataFrame(y_train.reset_index(drop=True))], axis=1)
        numericalData = pd.concat(
            [pd.DataFrame(xNumerical.reset_index(drop=True)), pd.DataFrame(y_train.reset_index(drop=True))], axis=1)
        # data = pd.concat(
        #     [pd.DataFrame(x_train.reset_index(drop=True)), pd.DataFrame(y_train.reset_index(drop=True))], axis=1)

        xNumerical = featureSelectionNumerical(numericalData, xNumerical)

        xCategorical = featureSelectionCategorical(categoricalData, xCategorical)

        x_train = pd.concat([pd.DataFrame(xCategorical.reset_index(drop=True)), pd.DataFrame(xNumerical.reset_index(drop=True))],axis=1)

        # print(x_train.columns)
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
        topFeatureNumerical = saveLoad.loadModel('topFeatureNumericalRegression')
        topFeaturesCategorical = saveLoad.loadModel('topFeaturesCategoricalRegression')

        xTest = pd.concat([pd.DataFrame(xTest[topFeatureNumerical]), pd.DataFrame(xTest[topFeaturesCategorical])], axis=1)

        polyPredict, decisionTreePredict, rfPrediction, enetPrediction = testPredict.predictRegression(xTest)

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

        plot_regression_line('Polynomial Regression', polyPredict, y_test)
        plot_regression_line('Decision Tree Regression', decisionTreePredict, y_test)
        plot_regression_line('Random Forest Regression', rfPrediction, y_test)
        plot_regression_line('Elastic Regression', enetPrediction, y_test)

        # Plot model fit for each model
        plot_model_fit('Polynomial Regression', polyPredict, y_test)
        plot_model_fit('Decision Tree Regression', decisionTreePredict, y_test)
        plot_model_fit('Random Forest Regression', rfPrediction, y_test)
        plot_model_fit('Elastic Regression', enetPrediction, y_test)

    elif x == '2':

        testPredict = test.TestPredict()
        testPredict.readData('ms1-games-tas-test-v1.csv')
        test = testPredict.preprocess()
        # print(test.head())
        topFeatureNumerical = saveLoad.loadModel('topFeatureNumericalRegression')
        topFeaturesCategorical = saveLoad.loadModel('topFeaturesCategoricalRegression')

        test = pd.concat([pd.DataFrame(test[topFeatureNumerical]), pd.DataFrame(test[topFeaturesCategorical])],
                         axis=1)

        # print(test.head())

        prediction = testPredict.predictRegression(test)

        # pd.set_option('display.max_rows', None)
        # pd.set_option('display.max_columns', None)
        # pd.set_option('display.expand_frame_repr', False)

        prediction = list(prediction)  # Convert the tuple to a list
        prediction[0] = pd.DataFrame(prediction[0], columns=['Average User Rating'])
        prediction[1] = pd.DataFrame(prediction[1], columns=['Average User Rating'])
        prediction[2] = pd.DataFrame(prediction[2], columns=['Average User Rating'])
        prediction[3] = pd.DataFrame(prediction[3], columns=['Average User Rating'])
        prediction = tuple(prediction)  # Convert the list back to a tuple if needed

        print(f"Polynomial Regression:\n{prediction[0].head()}")
        print(f"Decision Regression:\n{prediction[1].head()}")
        print(f"RandomForest Regression:\n{prediction[2].head()}")
        print(f"Elastic Regression:\n{prediction[3].head()}")
