import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class MaximumLikelihood:
    def __init__(self, mean = 0, std = 1, datapoints = 100):
        self.mean = mean
        self.std = std
        self.distribution = pd.DataFrame()
        self.no_of_datapoints = datapoints
        self.gaussianDistribution(self.mean,self.std)

    def gaussianDistribution(self,mean,std):
        X = self.generateRandomValues()
        N = []
        for x in X:
         N.append((1/(2*np.pi*np.square(std)))*np.exp((-1/(2*np.square(std)))*np.square(x-mean)))
        self.distribution = pd.DataFrame({'X': X, 'Distribution': N})
        #return N
    def likelihood(self,mean,std):
        #X = self.generateRandomValues()
        likelihood = 1
        #N = []
        X = self.distribution.X
        for x in X:
            #N.append((1 / (2 * np.pi * np.square(self.std))) * np.exp(
            #    (-1 / (2 * np.square(self.std))) * np.square(x - self.mean)))
            likelihood *= (1 / (2 * np.pi * np.square(std))) * np.exp(
                (-1 / (2 * np.square(std))) * np.square(x - mean))
        #self.distribution = pd.DataFrame({'X': X, 'Distribution': N})
        return likelihood

    def maximumLikelihood(self):
        estimate_mean = np.random.uniform(low=-2, high=2, size=(100,))
        np.append(estimate_mean,0)
        std=2
        likelihood_output = []
        for m in estimate_mean:
            likelihood_output.append(self.likelihood(m,std))

        df = pd.DataFrame({'estimate_mean':estimate_mean,'likelihood':likelihood_output})
        max_likelihood_idx = df['likelihood'].idxmax()
        max_likelihood_value= df['likelihood'].max()
        mean_ML = df['estimate_mean'][max_likelihood_idx]
        df = df.sort_values(by='estimate_mean', ignore_index=True)

        ###Plot Graph######
        fig_1, axes_1 = plt.subplots(figsize=(8,4),nrows=1, ncols=3)

        ####Plot Original Distribution ############
        dist_df=self.distribution.sort_values(by='X',ignore_index=True)
        axes_1[0].plot(dist_df.X, dist_df.Distribution,color='navy', label='Original Plot')
        axes_1[0].text(dist_df['X'][0], dist_df.sort_values(by='Distribution')['Distribution'].iloc[-2], r'$\mu = 0$' + "\n"+r'$\sigma = 2$',
                       color='navy')
        axes_1[0].set_title('Distribution')

        ####Plot for maximization for mean ############
        axes_1[1].plot(df.estimate_mean,df.likelihood,marker='o', markersize=7, markerfacecolor='orange',
                    markeredgecolor='yellow', markeredgewidth=4)
        axes_1[1].text(df['estimate_mean'][1],max_likelihood_value,r'$\mu_{ML} = $'+str(mean_ML)+'    '+ r'$\sigma = 2$',color='black')
        axes_1[1].set_title('Maximum Likehood for mean')

        ####Calculate maximum likelihood of std ############
        estimate_std = np.random.uniform(low=0, high=2, size=(100,))
        np.append(estimate_std, 1)
        #std = 2
        likelihood_output = []
        for s in estimate_std:
            likelihood_output.append(self.likelihood(mean_ML, s))

        df_std = pd.DataFrame({'estimate_std': estimate_mean, 'likelihood': likelihood_output})
        max_likelihood_idx = df_std['likelihood'].idxmax()
        max_likelihood_value = df_std['likelihood'].max()
        std_ML = df_std['estimate_std'][max_likelihood_idx]
        df_std = df_std.sort_values(by='estimate_std',ignore_index=True)

        ####Plot for maximization for std ############

        axes_1[2].plot(df_std.estimate_std, df_std.likelihood, marker='o', markersize=7, markerfacecolor='orange',
                       markeredgecolor='yellow', markeredgewidth=4)
        axes_1[2].text(df_std['estimate_std'][0], max_likelihood_value,r'$\mu_{ML} = $' + str(mean_ML) +"\n", color='black')
        y_cord = df_std.sort_values(by='likelihood')['likelihood'].iloc[-2]
        axes_1[2].text(df_std['estimate_std'][0], max_likelihood_value,
                       r'$\sigma_{ML} = $' + str(std_ML), color='black')
        axes_1[2].set_title('Maximum Likehood for std')

        ####Plot Distribution after maximization ############
        self.gaussianDistribution(mean_ML, std_ML)
        dist_df = self.distribution.sort_values(by='X', ignore_index=True)
        axes_1[0].plot(dist_df.X, dist_df.Distribution, color='orange',label='With Maximum Likelihood')
        axes_1[0].text(dist_df['X'][0], dist_df.sort_values(by='Distribution')['Distribution'].iloc[-2],
                       r'$\mu = $' +str(mean_ML)+ "\n" + r'$\sigma = $'+str(std_ML),
                       color='orange')
        axes_1[0].legend(loc=0)

        plt.show()


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
#ml.gaussianDistribution()
#ml.plotGraph()
ml.maximumLikelihood()