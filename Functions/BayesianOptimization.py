def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import sklearn.gaussian_process as skg
import sklearn.neural_network as sknn
import sklearn.ensemble as sken
from sklearn.model_selection import GridSearchCV


import numpy as np
from scipy import optimize


from Functions import Plotting
from Surrogates import nDimensionalFunctions as ND

def randomX(dim):
    randarray = np.random.rand(dim)
    return [ii for ii in randarray]

class run:
    def __init__(self, modeltype='GPR', policy='UCB', surrogate='ackley', noise=0, dimensions=2, runlength = 25,
                 folderpath='', startRandSamples=5):
        self.modeltype = modeltype
        self.policy = policy
        self.surrogate = surrogate
        self.runlength = runlength
        self.filepath = folderpath + '\\Opt ' + surrogate + 'N' + str(noise) + '_' + modeltype + '_' + policy
        self.nStart = startRandSamples
        self.dim = dimensions
        self.noise = noise

        self.surrfunc = getattr(ND, surrogate)
        #self.X = []
        #self.Y = []
    def buildStartRand(self):
        self.X = []
        self.Y = []
        self.MSE = []

        for ii in range(self.nStart):
            tempX = randomX(self.dim)
            self.X.append(tempX)

            tempY = self.surrfunc(tempX, self.noise)
            self.Y.append(tempY)

            self.MSE.append(0)

    def singleOptimization(self):
        self.buildStartRand()

        while len(self.Y) < self.runlength:
            #print('what?')
            self.model = BeliefModels(self.X, self.Y).modelPicker(self.modeltype)
            tempX = Minimization(self.model, self.modeltype, self.policy, self.dim).basefminSearch()
            tempY = self.surrfunc(tempX, self.noise)
            tempMSE = self.getMSE()

            self.X.append(tempX)
            self.Y.append(tempY)
            print(len(self.Y))
            self.MSE.append(tempMSE)

        return self

    def dimensionScreen(self, dimensions=[2, 3, 4]):
        Y = []
        for dim in dimensions:
            self.dim = dim
            tempY = self.singleOptimization()
            Y.append(tempY)
        Plotting.saveDimensionLinePlots(Y, dimensions, self.surrogate, self.filepath)

    def getMSE(self, nTestSamples=100):
        if self.modeltype == 'RND':
            return 0

        # testX = []
        # actualY = []
        # predictedY = []
        SE = []
        for ii in range(nTestSamples):
            tempX = randomX(self.dim)
            # testX.append(tempX)

            tempYact = self.surrfunc(tempX, self.noise)
            # actualY.append(tempYact)

            tempYpred, tempYErr = Prediction(tempX, self.model).predictYYErr(self.modeltype)
            # predictedY.append(tempYpred)

            SE.append((tempYpred-tempYact) ** 2)
        return np.mean(SE)


class BeliefModels:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.dim = len(X[0])

    def modelPicker(self, modeltype):
        if modeltype == 'RND':
            model = []
        elif modeltype == 'GPR':
            model = self.gaussianProcessRegression()
        elif modeltype == 'BRMLPR_EGS':
            best_estimator = self.model_mLPRegressionExhaustiveGridSearch()
            model = self.baggingRegressor(estimator=best_estimator)
        return model

    def gaussianProcessRegression(self):
        model = skg.GaussianProcessRegressor()
        model.fit(self.X, self.Y)
        return model


    def model_mLPRegressionExhaustiveGridSearch(self):
        param_grid = {"activation": ["identity", "logistic", "tanh", "relu"],
                      "solver": ["lbfgs"], "alpha": [a for a in np.logspace(-6, 0, 3)]}
        GSmodel = GridSearchCV(sknn.MLPRegressor(), param_grid)
        GSmodel.fit(self.X, self.Y)
        model = GSmodel.best_estimator_
        return model

    def baggingRegressor(self, estimator):
        self.estimator=estimator
        model = sken.BaggingRegressor(estimator=estimator)
        model.fit(self.X, self.Y)
        return model


class DecisionPolicies:
    def __init__(self, model, modeltype, policy):
        self.model = model
        self.modeltype = modeltype
        self.policy = policy

    def policyPicker(self, X):
        if self.policy == 'UCB':
            self.Y, self.YErr = Prediction(X, self.model).predictYYErr(self.modeltype)
            value = self.upperConfidenceBounds()
        return -value

    def upperConfidenceBounds(self):
        lam = 1 / (2 ** 0.5)
        value = self.Y + lam * self.YErr
        return value

class Prediction:
    def __init__(self, X, model):
        self.X = X
        self.model = model

    def predictYYErr(self, modeltype):
        self.modeltype = modeltype
        if self.modeltype in ['GPR']:
            result = self.model.predict([self.X], return_std=True)
            Y = result[0]
            YErr = result[1]

        elif self.modeltype in ['BRMLPR_EGS']:
            result = self.model.predict([self.X])
            Y = result[0]
            resultmembers = [x.predict([self.X]) for x in self.model.estimators_]
            YErr = np.std(resultmembers)

        return Y, YErr

class Minimization:
    def __init__(self, model, modeltype, policy, dim):
        self.model = model
        self.modeltype = modeltype
        self.policy = policy
        self.dim = dim

    def basefminSearch(self):
        if self.modeltype == 'RND':
            return np.random.rand(self.dim)

        f = lambda X: DecisionPolicies(self.model, self.modeltype, self.policy).policyPicker(X)

        bnds = []
        for ii in range(self.dim):
            bnds.append((0, 1))

        result = optimize.minimize(f, np.random.rand(self.dim), method='Nelder-Mead', bounds=bnds)
        newX = result.x
        return newX