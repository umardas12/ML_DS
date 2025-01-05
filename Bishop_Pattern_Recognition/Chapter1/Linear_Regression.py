#solving the linear regression of chapter 1, Bishop's Pattern Recognition


import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression:

   def __init__(self,order_of_polynomial,learning_rate,datapoints,convergence=1e-6):
      self.convergence = convergence
      self.alpha_learning_rate = learning_rate
      self.m_polynomial = order_of_polynomial
      self.weight = np.random.randn(self.m_polynomial+1) * 0.01 #[np.zeros(features)]
      self.datapoints = datapoints
      self.input_x = []
      self.observed_target_t = []
      self.polynomial = np.full([self.m_polynomial+1,self.datapoints],1.0000000000000)
      self.training_dataset = pd.DataFrame()
      self.createTrainingDataset(self.datapoints)

   def constructPolynomial(self):
      print(self.input_x)
      for i in range(1,self.m_polynomial+1):
         self.polynomial[i]=np.power(self.input_x,i)
      print(f'polynomial: {self.polynomial}',file= open('LinearRegression_output.txt','w'))

   def leastSquareError(self,prediction):
      E = (np.dot(prediction - self.observed_target_t,prediction - self.observed_target_t))
      avgE = E/self.datapoints
      print(f'leastSquareError: {E}',file= open('LinearRegression_output.txt','a'))
      return avgE

   def gradientDescent(self,prediction):
      dW = np.dot(prediction-self.observed_target_t,self.polynomial.transpose())
      self.weight = self.weight - self.alpha_learning_rate * dW
      print(f'weight: {self.weight}',file= open('LinearRegression_output.txt','a'))

   def predictionCalculation(self):
      prediction = np.dot(self.weight,self.polynomial)
      print (f'prediction: {prediction}',file= open('LinearRegression_output.txt','a'))
      return prediction

   def optimiseWeight(self):
      self.constructPolynomial()
      iteration = 0
      print(f'==================================================', file=open('LinearRegression_output.txt', 'a'))
      print(f'iteration: {iteration}',file= open('LinearRegression_output.txt','a'))
      Y = self.predictionCalculation()
      E = self.leastSquareError(Y)
      E_prev = 0
      while abs(E-E_prev)>self.convergence:
         self.gradientDescent(Y)
         Y = self.predictionCalculation()
         E_prev = E
         E = self.leastSquareError(Y)
         iteration += 1
         print(f'==================================================', file=open('LinearRegression_output.txt', 'a'))
         print(f'iteration: {iteration}',file=open('LinearRegression_output.txt', 'a'))
         if (iteration == 50000):
            break
      print(f'Total Iterations Taken:{iteration}',file=open('LinearRegression_outputsummary.txt', 'w'))
      print(f'LSE:{E}',file=open('LinearRegression_outputsummary.txt', 'a'))
      print(f'PolynomialOrder: {self.m_polynomial}', file=open('LinearRegression_outputsummary.txt', 'a'))
      print(f'LearningRate: {self.alpha_learning_rate}',file=open('LinearRegression_outputsummary.txt', 'a'))
      print(f'weight: {self.weight}',file=open('LinearRegression_outputsummary.txt', 'a'))

      plt.scatter(self.input_x,Y,color = 'orange')
      plt.scatter(self.input_x,self.observed_target_t, color = 'blue')
      plt.show()


   def createTrainingDataset(self,no_of_datapoints):
      pi = np.pi
      #input_x = []
      actual_output_y = []
      #observed_target_t = []
      noise = np.random.normal(0,1,no_of_datapoints+1)
      for i in range(0,no_of_datapoints):
         self.input_x.append( random.random()) #random() generates a float between 0 and 1
         actual_output_y.append(np.sin(2*pi*self.input_x[i]))
         self.observed_target_t.append(actual_output_y[i]+0.2*noise[i])

      self.training_dataset = pd.DataFrame({'Input_X': self.input_x, 'ActualTarget': actual_output_y, 'ObservedTarget': self.observed_target_t})
      #print(f'target: {self.observed_target_t}',file=open('LinearRegression_outputsummary.txt', 'w'))

      #plt.scatter(input_x, actual_output_y)
      #plt.scatter(input_x,observed_target_t)
      #plt.show()



lr = LinearRegression(order_of_polynomial=3,learning_rate=0.01,datapoints=100)
lr.optimiseWeight()


