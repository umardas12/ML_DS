import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math



class MyLogitRegression(object):

    def __init__(self,X,Y,eta=0.01,iteration=100,lambda_reg=0.0001,nhlyr=3,nhlyrel):

        self.m_sample=X.shape[0]
        self.inv_m_sample=1.0/self.m_sample
        #self.dimension=X.shape[1]
        self.n_iter=iteration
        self._lambda=lambda_reg
        self.n_hidden_layer=nhlyr
        self.n_layer_element=nhlyrel
        

    def initializeNetwork(self,):
        self.hidden_layer=[]
        for i in range (self.n_hidden_layer+1):
            l={'weights':[[0.0 for j in range(self.n_layer_element[i]+1)] for j in range(self.n_layer_element[i+1])]}
            self.hidden_layer.append(l)
        #output_layer=[{'weights':[0.0 for j in range(self.nhlyrel[-2]+1)]}]
                
        

    def estimateTheta(self,X,Y):
        #gradient descent
        for i in range(self.n_iter):
            for xi in range(self.m_sample): 
                for li,l in enumerate(self.hidden_layer):
                    for k in len(l['weights']):
                        l['weights'][k] -= self._eta*delta_theta(X[xi],Y)*X[xi]
        

    def deltaTheta(self,X,Y):
        self.backPropagation(X,Y)

    def forwardPropagation(self,l):
        ip=np.array(activation[-1])
        theta=np.array(self.hidden_layer[l]['weights'])
        z=np.dot(theta[1:],ip.T)+theta[0]
        activate=self.sigmoid(z)
        self.activation.append(activate)
               
        
    
    def backPropagation(self,X,Y ):

        #initialize delta theta
        delta=[]
        for i in range (self.n_hidden_layer+1):
            d={'delta':[[0.0 for j in range(self.n_layer_element[i]+1)] for j in range(self.n_layer_element[i+1])]}
            delta.append(d)
            
        
        #calculate activation in each layer
        self.activation=[]
        l=[X[j].values for j in range(len(X))]
        self.activation.append(l)
        for l in range(1,self.n_layer_element):
            self.forwardPropagation(l)
                
        #compute error delta for each layer
        #errors=[]
            
        for k in reversed(range(1,len(self.n_layer_element))):
            if k!=len(n_layer_element)-1:
                for e in range(self.n_layer_element[k]):
                    error=np.dot(theta[k].T,errors[1])*(activation[k]*(1-activation[k]))
                    delta[k-1]+=np.dot(error,activation[k-1].T)
                    delta[k-1]*=self.inv_m_sample
                    #errors.append(error)

            else:
                error=activation[-1]-Y
                delta[k-1]+=np.dot(error,activation[k-1].T)
                delta[k-1]*=self.inv_m_sample
                #errors.append(error)
        return delta       
            
        
    def sigmoid(self,z):
        return (1.0/(1.0+np.exp(-z)))

        
        

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
