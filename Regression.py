from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
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

        # Fit the Lasso regression model
        lasso = Lasso(alpha=0.0000001)
        lasso.fit(self.X, self.Y)
        self.saveLoad.saveModel(lasso, 'lassoModel')
        y_lasso = lasso.predict(self.X)

        # Fit the support vector regression model
        svr = SVR(kernel='rbf', C=1e3, gamma=0.1)
        svr.fit(self.X, self.Y)
        self.saveLoad.saveModel(svr, 'SVR')
        svrPrediction = svr.predict(self.X)

        # Fit the elastic net regression model
        alpha = 0.1  # regularization strength
        l1_ratio = 0.5  # balance between L1 and L2 regularization
        enet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        enet.fit(self.X, self.Y)
        self.saveLoad.saveModel(enet, 'elastic')
        enetPrediction = enet.predict(self.X)

        # Create a decision tree regressor with maximum depth of 3
        regressor = DecisionTreeRegressor(max_depth=6)
        # Fit the regressor to the training data
        regressor.fit(self.X, self.Y)
        # Save DecisionTreeRegressorModel
        self.saveLoad.saveModel(regressor, 'DecisionTreeRegressorModel')
        # Make predictions on the test set
        y_decision = regressor.predict(self.X)

        return y_poly, y_lasso, y_decision, rfPrediction, svrPrediction, enetPrediction

    def test(self):
        # Load models
        polyModel = self.saveLoad.loadModel('LinearModel')
        poly = self.saveLoad.loadModel('PolynomialModel')
        lasso = self.saveLoad.loadModel('lassoModel')
        decisionTree = self.saveLoad.loadModel('DecisionTreeRegressorModel')
        svrModel = self.saveLoad.loadModel('SVR')
        enet = self.saveLoad.loadModel('elastic')
        rf = self.saveLoad.loadModel('RandomForestRegressorModel')

        # Transform the test data to include polynomial features
        xtest_poly = poly.transform(self.X)

        # Use the model to predict X
        polyPredict = polyModel.predict(xtest_poly)
        lassoPredict = lasso.predict(self.X)
        decisionTreePredict = decisionTree.predict(self.X)
        enetPrediction = enet.predict(self.X)
        rfPrediction = rf.predict(self.X)
        svrPrediction = svrModel.predict(self.X)

        return polyPredict, lassoPredict, decisionTreePredict, rfPrediction, svrPrediction, enetPrediction

