import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif, SelectKBest
import Classification
import Preprocessing
import saveLoadData
import test
from sklearn.metrics import accuracy_score

def preprocess_input(filename):

    # Read data from CSV file to dataset
    dataset = pd.read_csv(filename)
    # Split all column except last one for input in X
    X = dataset.iloc[:,:-1]
    # Split last column for output in Y
    Y = pd.DataFrame(dataset.iloc[:,-1])

    return X, Y

# Enter your code her
def featureSelectionNumerical(data, input):
    # Use the ANOVA F-test to select the top k features
    selector = SelectKBest(f_classif, k=5)
    inputData = selector.fit_transform(input, data['Rate'])

    # Get the selected feature indices
    feature_indices = selector.get_support(indices=True)

    # Get the feature names and ANOVA F-scores
    feature_names = input.columns[feature_indices]
    # f_scores = selector.scores_[feature_indices]

    inputData = pd.DataFrame(inputData, columns=feature_names)

    saveLoad.saveModel(feature_names, 'topFeatureNumericalClassification')
    # Print feature importance scores
    # for feature, score in zip(feature_names, f_scores):
    #     print(f"{feature}: {score}")

    return inputData

def featureSelectionCategorical(data, input):
    # Use the chi-squared test to select the top k features
    selector = SelectKBest(chi2, k=5)
    inputData = selector.fit_transform(input, data['Rate'])

    # Get the selected feature indices
    feature_indices = selector.get_support(indices=True)

    # Get the feature names and chi-squared statistics
    feature_names = input.columns[feature_indices]
    # chi2_scores = selector.scores_[feature_indices]

    inputData = pd.DataFrame(inputData, columns=feature_names)

    saveLoad.saveModel(feature_names, 'topFeaturesCategoricalClassification')
    # Print feature importance scores
    # for feature, score in zip(feature_names, chi2_scores):
    #     print(f"{feature}: {score}")

    return inputData

def calcAccuracy(modelPred, yTest , modelName):

    score_model = accuracy_score(yTest, modelPred)
    print(f"accuracy score of {modelName} model : {score_model}")

if __name__=='__main__':
    x = input("Enter 1 if you want run train , 2 for test : ")

    # Create object to load/save data/models in/from file
    saveLoad = saveLoadData.SaveLoadData()
    if x == '1':
        # Get input in X & output in Y
        X, Y = preprocess_input('games-classification-dataset.csv')

        # Splitting the X,Y into the Training set(80%) and Test set(20%)
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

        # Create object of Preprocess Class
        preprocess = Preprocessing.Preprocess(x_train, y_train)

        # Clean Training Data
        x_train, y_train = preprocess.trainDataCleaning()
        # pd.set_option('display.max_rows', None)
        # pd.set_option('display.max_columns', None)
        # pd.set_option('display.expand_frame_repr', False)
        #
        # doc2vec_cols = x_train.filter(regex='^doc2vec_')
        # x_train = x_train.copy()
        # xNumerical = x_train.iloc[:, 0:5]
        # tmpList = ['Developer', 'Age Rating']
        # for col in tmpList:
        #     xNumerical.drop(col, axis=1, inplace=True)
        # xNumerical = pd.concat([pd.DataFrame(xNumerical.reset_index(drop=True)), pd.DataFrame(doc2vec_cols.reset_index(drop=True))], axis=1)
        #
        # doc2vecNames = doc2vec_cols.columns
        # xCategorical = x_train.iloc[:, 5:]
        # xCategorical['Developer'] = x_train['Developer']
        # xCategorical['Age Rating'] = x_train['Age Rating']
        #
        # xCategorical = xCategorical.drop(doc2vecNames, axis=1)
        #
        # # Concatenate x_train and y_train in data
        # categoricalData = pd.concat(
        #     [pd.DataFrame(xCategorical.reset_index(drop=True)), pd.DataFrame(y_train.reset_index(drop=True))], axis=1)
        # numericalData = pd.concat(
        #     [pd.DataFrame(xNumerical.reset_index(drop=True)), pd.DataFrame(y_train.reset_index(drop=True))], axis=1)
        #
        # xNumerical = featureSelectionNumerical(numericalData, xNumerical)
        #
        # xCategorical = featureSelectionCategorical(categoricalData, xCategorical)
        #
        # x_train = pd.concat(
        #     [pd.DataFrame(xCategorical.reset_index(drop=True)), pd.DataFrame(xNumerical.reset_index(drop=True))], axis=1)
        #
        # print("===================================================")
        # classification = Classification.classification(x_train, y_train)
        # predictions = classification.trainData()
        # calcAccuracy(predictions[0], y_train, "RandomForestClassifier")
        # calcAccuracy(predictions[1], y_train, "KNeighborsClassifier")
        # calcAccuracy(predictions[2], y_train, "GradientBoostingClassifier")
        # calcAccuracy(predictions[3], y_train, "DecisionTreeClassifier")
        # calcAccuracy(predictions[4], y_train, "LogisticRegression")
        # calcAccuracy(predictions[5], y_train, "SupportVectorMachineClassifier")
        #
        # testPredict = test.TestPredict(x_test)
        # xTest = testPredict.preprocess()
        # topFeatureNumerical = saveLoad.loadModel('topFeatureNumericalClassification')
        # topFeaturesCategorical = saveLoad.loadModel('topFeaturesCategoricalClassification')
        #
        # xTest = pd.concat([pd.DataFrame(xTest[topFeatureNumerical]), pd.DataFrame(xTest[topFeaturesCategorical])], axis=1)
        #
        # predictions = testPredict.predictClassification(xTest)
        # print("===========================================Test============================================================")
        # calcAccuracy(predictions[0], y_test, "RandomForestClassifier")
        # calcAccuracy(predictions[1], y_test, "KNeighborsClassifier")
        # calcAccuracy(predictions[2], y_test, "GradientBoostingClassifier")
        # calcAccuracy(predictions[3], y_test, "DecisionTreeClassifier")
        # calcAccuracy(predictions[4], y_test, "LogisticRegression")
        # calcAccuracy(predictions[5], y_test, "SupportVectorMachineClassifier")

    elif x == '2':

        testPredict = test.TestPredict()
        testPredict.readData('ms2-games-tas-test-v1.csv')
        test = testPredict.preprocess()

        topFeatureNumerical1 = saveLoad.loadModel('topFeatureNumericalClassification')
        topFeaturesCategorical1 = saveLoad.loadModel('topFeaturesCategoricalClassification')
        # print(test.head())
        # print(topFeatureNumerical1)
        # print(topFeaturesCategorical1)
        test = pd.concat([pd.DataFrame(test[topFeatureNumerical1]), pd.DataFrame(test[topFeaturesCategorical1])], axis=1)

        # print(test.head())

        prediction = testPredict.predictClassification(test)

        # pd.set_option('display.max_rows', None)
        # pd.set_option('display.max_columns', None)
        # pd.set_option('display.expand_frame_repr', False)

        prediction = list(prediction)  # Convert the tuple to a list
        prediction[0] = pd.DataFrame(prediction[0], columns=['Rate'])
        prediction[1] = pd.DataFrame(prediction[1], columns=['Rate'])
        prediction[2] = pd.DataFrame(prediction[2], columns=['Rate'])
        prediction[3] = pd.DataFrame(prediction[3], columns=['Rate'])
        prediction[4] = pd.DataFrame(prediction[4], columns=['Rate'])
        prediction[5] = pd.DataFrame(prediction[5], columns=['Rate'])
        prediction = tuple(prediction)  # Convert the list back to a tuple if needed

        print(f"RandomForestClassifier:\n{prediction[0].head()}")
        print(f"KNeighborsClassifier:\n{prediction[1].head()}")
        print(f"GradientBoostingClassifier:\n{prediction[2].head()}")
        print(f"DecisionTreeClassifier:\n{prediction[3].head()}")
        print(f"LogisticRegression:\n{prediction[4].head()}")
        print(f"SupportVectorMachineClassifier:\n{prediction[5].head()}")