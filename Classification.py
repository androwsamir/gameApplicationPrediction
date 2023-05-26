from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import time
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.svm import SVC
import saveLoadData


class classification:

    saveLoad = saveLoadData.SaveLoadData()

    # Constructor
    def __init__(self, X, Y=None):
        self.X = X
        self.Y = Y

    def trainData(self):

        c = 0.0000000000000001

        rf = RandomForestClassifier(ccp_alpha=0.1)
        knn = KNeighborsClassifier(n_neighbors=1000)
        gb = GradientBoostingClassifier(learning_rate=0.00001)
        dt = DecisionTreeClassifier(ccp_alpha=0.1, criterion='entropy')
        logreg = LogisticRegression(C=c, solver='saga')
        svm = SVC(C=c, kernel='rbf')

        rf_begin = time.time()
        rf.fit(self.X, np.ravel(self.Y))
        rf_end = time.time()
        #####################
        knn_begin = time.time()
        knn.fit(self.X, np.ravel(self.Y))
        knn_end = time.time()
        #####################
        gb_begin = time.time()
        gb.fit(self.X, np.ravel(self.Y))
        gb_end = time.time()
        #####################
        dt_begin = time.time()
        dt.fit(self.X, np.ravel(self.Y))
        dt_end = time.time()
        #####################
        logreg_begin = time.time()
        logreg.fit(self.X, np.ravel(self.Y))
        logreg_end = time.time()
        #####################
        sv_begin = time.time()
        svm.fit(self.X, np.ravel(self.Y))
        sv_end = time.time()

        index = ['RandomForestClassifier',
                 'KNeighborsClassifier',
                 'GradientBoostingClassifier',
                 'DecisionTreeClassifier',
                 'LogisticRegression',
                 'SupportVectorMachineClassifier']
        data = [rf_end - rf_begin, knn_end - knn_begin,
                gb_end - gb_begin, dt_end - dt_begin,
                logreg_end - logreg_begin, sv_end - sv_begin]
        fig = plt.Figure(figsize=(10, 40))
        plt.bar(index, data)
        plt.xlabel('Models')
        plt.ylabel('time values')
        plt.title('Models tranning time')
        plt.show()

        self.saveLoad.saveModel(rf, 'RandomForestClassifier')
        self.saveLoad.saveModel(knn, 'KNeighborsClassifier')
        self.saveLoad.saveModel(gb, 'GradientBoostingClassifier')
        self.saveLoad.saveModel(dt, 'DecisionTreeClassifier')
        self.saveLoad.saveModel(logreg, 'LogisticRegression')
        self.saveLoad.saveModel(svm, 'SupportVectorMachineClassifier')

        return rf.predict(self.X), knn.predict(self.X), gb.predict(self.X), dt.predict(self.X), logreg.predict(
            self.X), svm.predict(self.X)

    def testData(self):

        rf = self.saveLoad.loadModel('RandomForestClassifier')
        knn = self.saveLoad.loadModel('KNeighborsClassifier')
        gb = self.saveLoad.loadModel('GradientBoostingClassifier')
        dt = self.saveLoad.loadModel('DecisionTreeClassifier')
        logreg = self.saveLoad.loadModel('LogisticRegression')
        svm = self.saveLoad.loadModel('SupportVectorMachineClassifier')

        rfp_begin = time.time()
        rf_pr = rf.predict(self.X)
        rfp_end = time.time()
        ##########
        knp_begin = time.time()
        knn_pr =knn.predict(self.X)
        knp_end = time.time()
        ##############
        gbp_begin = time.time()
        gb_pr = gb.predict(self.X)
        gbp_end = time.time()
        ##############
        dtp_begin = time.time()
        dt_pr = dt.predict(self.X)
        dtp_end = time.time()
        ##############
        lrp_begin = time.time()
        lr_pr = logreg.predict(self.X)
        lrp_end = time.time()
        ##############
        svp_begin = time.time()
        sv_pr = svm.predict(self.X)
        svp_end = time.time()
        ##############
        index = ['RandomForestClassifier',
                 'KNeighborsClassifier',
                 'GradientBoostingClassifier',
                 'DecisionTreeClassifier',
                 'LogisticRegression',
                 'SupportVectorMachineClassifier']
        data = [rfp_end - rfp_begin, knp_end - knp_begin,
                gbp_end - gbp_begin, dtp_end - dtp_begin,
                lrp_end - lrp_begin, svp_end - svp_begin]
        fig2 = plt.Figure(figsize=(10, 40))
        plt.bar(index, data)
        plt.xlabel('Models')
        plt.ylabel('time values')
        plt.title('Models Testing Time')
        plt.show()

        return rf_pr, knn_pr, gb_pr, dt_pr, lr_pr, sv_pr
