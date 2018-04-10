import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math



class MyLogitRegression(object):

    def __init__(self,X,Y,eta=0.01,iteration=100,lambda_reg=0.0001):
        self.W = (np.zeros(len(X.columns)+1))
        self._eta=eta
        self.n_iter=iteration
        self.m_sample=X.shape[0]
        self.inv_m=1.0/self.m_sample
        self.cost=[]
        self._lambda=lambda_reg


    def estimateW(self,X,Y):
        self.cost=[]
        for i in range(self.n_iter):
            #print("estimate",i)
            self.gradientDescent(X,Y)
	    #Calculate costFunction for the estimate
            self.costFunction(X,Y)
        #plot costFunction
        self.plotCostFunction(self.cost,self.n_iter)
        self.plotXY(X,Y)
            
    def gradientDescent(self,X,Y):
        #print("gradient",type(X))
        prob=self.hypothesis(X)
        errors=(prob-Y.T)#errors is a row vector; it is the common term of CostFunction Derivative
        iterm=self.inv_m*self._eta*errors
        r=(1-self._eta*self._lambda*self.inv_m)#reduce weight by regularisation term
        self.W=self.W*r
        self.W[0]-=iterm.sum()
        self.W[1:]-=np.dot(iterm,X)
        
    
    def predict(self,X):
        return np.where(self.hypothesis(X)>=0.5,1,0)
    
    def hypothesis(self,X):
        #print("hypothesis",type(X))
        z=np.dot(self.W[1:].T,X.T)+self.W[0] #W is 1 D array;Z is a row vector
        return self.sigmoid(z)

    def sigmoid(self, z):
        G_of_z=1.0/(1.0+np.exp(-z))
        return G_of_z

    def costFunction(self,X,Y):
        costs=0
        reg_term=self.regularisationTerm()
        for i in range(self.m_sample):
            if Y[i]==0:
                costs-=math.log(1-self.hypothesis(X.iloc[i].values))+reg_term
            else:
                costs-=math.log(self.hypothesis(X.iloc[i].values))+reg_term
        costs*=self.inv_m #normalize cost
        self.cost.append(costs)

    def regularisationTerm(self):
        return (np.dot(self.W.T,self.W))*self._lambda

        
    def plotCostFunction(self,cost,Y):
        xx=np.arange(self.n_iter)
        plt.plot(xx,cost,linestyle='--',color='b',marker='x')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.title('Eta = %f'%(self._eta))
        plt.show()
        
    def plotXY(self,X,Y):
        xx=np.arange(50)
        y_pred=self.predict(X)
        plt.plot(xx,Y[:50],linestyle='--',color='b',marker='x')
        plt.plot(xx,y_pred[:50],linestyle='--',color='r',marker='o')
        plt.xlabel('Samples')
        plt.ylabel('prediction')
        plt.title('sample vs predition')
        plt.show()
        

'''df=pd.read_csv('C:/Users/uma/Desktop/Python_exp/data/classification/bc_data.csv',
               names=['Sample_ID','Clump_Thickness','Unif_of_cell_sz','Unif_of_cell_shape',
                      'Marginal_Adhesion','Ep_cell_sz','Bare_nuclei','Bland_Chromatin','Nucleoli','Mitoses','Class'])
df.replace('?',0,inplace=True)
df['Bare_nuclei']=df['Bare_nuclei'].astype('int64')
Y=df['Class'].values
X=df.iloc[:,1:10]
normalize_X=((X-X.mean())/X.std()) #normalized each column
# have taken all samples
O=MyLogitRegression(X,Y)
O.estimateW(X,Y,eta=1.0,iteration=100)'''


'''df=pd.read_csv('C:/Users/uma/Desktop/Python_exp/data/classification/iris.csv',
               names=['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width','Class'])
df_sample=df.sample(frac=1)
c=df_sample['Class'].values
Y=(c=='Iris-setosa').astype(int)
X=df_sample.iloc[:,0:4]
#train,test
Y_train=Y[:90]
Y_test=Y[90:]
X_train=X.iloc[:90,:]
X_test=X.iloc[90:,:]
meanx=X_train.mean()
stdx=X_train.std()
#normalize_X=((X-X.mean())/X.std()) #normalized each column
X_train=(X_train-meanx)/stdx
X_test=(X_test-meanx)/stdx
# have taken all samples
O=MyLogitRegression(X_train,Y_train,eta=0.1,iteration=100)
O.estimateW(X,Y)
print(Y_test[:])
y_pred=O.predict(X_test.iloc[:,:])
print(y_pred)'''



df=pd.read_csv('C:/Users/uma/Desktop/Python_exp/data/classification/bc_data.csv',
               names=['Sample_ID','Clump_Thickness','Unif_of_cell_sz','Unif_of_cell_shape',
                      'Marginal_Adhesion','Ep_cell_sz','Bare_nuclei','Bland_Chromatin','Nucleoli','Mitoses','Class'])
df.replace('?',0,inplace=True)
df['Bare_nuclei']=df['Bare_nuclei'].astype('int64')
c=df['Class'].values
Y=(c==4).astype(int)
X=df.iloc[:,1:10]
#train,test
Y_train=Y[:500]
Y_test=Y[500:]
X_train=X.iloc[:500,:]
X_test=X.iloc[500:,:]
meanx=X_train.mean()
stdx=X_train.std()
#normalize_X=((X-X.mean())/X.std()) #normalized each column
X_train=(X_train-meanx)/stdx
X_test=(X_test-meanx)/stdx
# have taken all samples
#O=MyLogitRegression(X_train,Y_train,eta=0.001,iteration=100,lambda_reg=0.01) #misclassifcation 2
O=MyLogitRegression(X_train,Y_train,eta=0.5,iteration=100,lambda_reg=0) #misclassifcation 2

O.estimateW(X_train,Y_train)
print(Y_test)
y_pred=O.predict(X_test)
print(y_pred)
O.plotXY(X_test,Y_test) 
