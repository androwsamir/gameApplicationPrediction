from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import  saveLoadData

class regression:

    saveLoad  = saveLoadData.SaveLoadData()

    def __init__(self, x, y=None):
        self.X = x
        self.Y = y

    def train(self):

        poly = PolynomialFeatures(degree=4)
        # Transform the data to include polynomial features
        x_poly = poly.fit_transform(self.X)
        poly_model = LinearRegression()
        poly_model.fit(x_poly, self.Y)

        # Save polynomial model
        self.saveLoad.saveModel(poly_model, 'LinearModel')
        self.saveLoad.saveModel(poly, 'PolynomialModel')

        # predicting on training data-set
        y_poly = poly_model.predict(x_poly)

        # Create a Random Forest regressor with 100 trees
        rf = RandomForestRegressor(n_estimators=100)
        # Fit the regressor to the training data
        rf.fit(self.X, self.Y)
        self.saveLoad.saveModel(rf, 'RandomForestRegressorModel')
        # Make predictions on the test set
        rfPrediction = rf.predict(self.X)

        # Fit the elastic net regression model
        alpha = 0.1  # regularization strength
        l1_ratio = 0.9  # balance between L1 and L2 regularization
        enet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        enet.fit(self.X, self.Y)
        self.saveLoad.saveModel(enet, 'elastic')
        enetPrediction = enet.predict(self.X)

        # Create a decision tree regressor with maximum depth of 3
        regressor = DecisionTreeRegressor(max_depth=3)
        # Fit the regressor to the training data
        regressor.fit(self.X, self.Y)
        # Save DecisionTreeRegressorModel
        self.saveLoad.saveModel(regressor, 'DecisionTreeRegressorModel')
        # Make predictions on the test set
        y_decision = regressor.predict(self.X)

        return y_poly, y_decision, rfPrediction, enetPrediction

    def test(self):
        # Load models
        polyModel = self.saveLoad.loadModel('LinearModel')
        poly = self.saveLoad.loadModel('PolynomialModel')
        decisionTree = self.saveLoad.loadModel('DecisionTreeRegressorModel')
        enet = self.saveLoad.loadModel('elastic')
        rf = self.saveLoad.loadModel('RandomForestRegressorModel')

        # Transform the test data to include polynomial features
        xtest_poly = poly.transform(self.X)

        # Use the model to predict X
        polyPredict = polyModel.predict(xtest_poly)
        decisionTreePredict = decisionTree.predict(self.X)
        enetPrediction = enet.predict(self.X)
        rfPrediction = rf.predict(self.X)

        return polyPredict, decisionTreePredict, rfPrediction, enetPrediction

