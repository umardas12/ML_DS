import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class MaximumLikelihood:
    def __init__(self, mean = 0, std = 1, datapoints = 100):
        self.mean = mean
        self.std = std
        self.distribution = pd.DataFrame()
        self.no_of_datapoints = datapoints
        #self.gaussianDistribution()

    def gaussianDistribution(self):
        X = self.generateRandomValues()
        N = []
        for x in X:
         N.append((1/(2*np.pi*np.square(self.std)))*np.exp((-1/(2*np.square(self.std)))*np.square(x-self.mean)))
        self.distribution = pd.DataFrame({'X': X, 'Distribution': N})
        #return N
    def likelihood(self):
        X = self.generateRandomValues()
        likelihood = 1
        N = []
        for x in X:
            #N.append((1 / (2 * np.pi * np.square(self.std))) * np.exp(
            #    (-1 / (2 * np.square(self.std))) * np.square(x - self.mean)))
            likelihood *= (1 / (2 * np.pi * np.square(self.std))) * np.exp(
                (-1 / (2 * np.square(self.std))) * np.square(x - self.mean))
        #self.distribution = pd.DataFrame({'X': X, 'Distribution': N})

    def maximumLikelihood(self):
        estimate_mean = np.random.uniform(high=3, low=-3, size(50,))


    def plotGraph(self):
        np_array = self.distribution.values
        x_values = np_array[:,0]
        y_values = np_array[:,1]
        fig_1, axes_1 = plt.subplots(figsize=(8,4),nrows=1, ncols=1)
        axes_1.scatter(x_values,y_values)
        plt.show()

    def generateRandomValues(self):
        V = np.random.uniform(low=-1, high=1, size=(100,))
        #for _ in range(self.no_of_datapoints+1):
        #    V.append(np.random.random())
        return V

ml = MaximumLikelihood()
ml.gaussianDistribution()
ml.plotGraph()