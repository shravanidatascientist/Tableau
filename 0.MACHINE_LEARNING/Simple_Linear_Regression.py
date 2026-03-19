import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
import streamlit as st

dataset = pd.read_csv(r"C:\Users\Shravani\Downloads\Salary_Data.csv")

x = dataset.iloc[:, :-1].values

y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state=0) 


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)


comparision = pd.DataFrame({'Actual': y_test, 'prediction' : y_pred})

plt.scatter(x_test, y_test, color = 'Red')
plt.plot(x_train, regressor.predict(x_train),color ='yellow')
plt.title('Salary Of The Employee Based On Experience')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

## Predicting the Future

m_coef = regressor.coef_
print(m_coef)

c_intercept = regressor.intercept_
print(c_intercept)

y_12 = m_coef*12 +c_intercept
print(y_12)

y_20 = m_coef*20 +c_intercept

## Training data ------ Bias

bias = regressor.score(x_train, y_train)
print(bias)

## Testingd data ------------ Variance

variance = regressor.score(x_test, y_test)
print(variance)


###  Stats implementation to this code 

import sys
print(sys.executable)


import scipy.stats as stats

dataset.apply(stats.zscore)


## SSR

y_mean = np.mean(y)

SSR = np.sum((y_pred- y_mean)**2)
print(SSR)

## SSE

y= y[0:6]
SSE = np.sum((y - y_pred)**2)
print(SSE)
## SST

mean_total = np.mean(dataset.values)
SST = np.sum((dataset.values-mean_total)**2)
print(SST)

## R2

r_square = 1 - SSR/SST
print(r_square)

import pickle

# Save the trained model to disk
filename = 'linear_regression_model.pkl'

# Open a file in write-binary mode and dump the model
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)

print("Model has been pickled and saved as linear_regression_model.pkl")




