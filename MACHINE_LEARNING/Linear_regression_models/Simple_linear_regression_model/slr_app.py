import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



dataset = pd.read_csv(r"C:\Users\shravani\Desktop\NIT\3rd, 4th  - SLR\3rd, 4th  - SLR\SIMPLE LINEAR REGRESSION\Salary_Data.csv")

## Let's separate independent variables and dependent variables 

X = dataset.iloc[:,:-1]

y = dataset.iloc[:,-1]


## splitting the dataset into training data and testing data using scikit-learn.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

## We import the Linear Regression algorithm, create the model, and train it using training data so it learns from the data.

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)


"""
Linear Regression is a machine learning algorithm that finds the best straight
line that fits the data. When we run regressor.fit(X_train, y_train), the model
learns the relationship between input variables and the output variable. It
calculates the difference between the actual value and predicted value, called
error, and tries to minimize this error using the Least Squares method.
"""

## This below code tells the trained model to predict the output values for the test input data.

y_pred = regressor.predict(X_test)

comparison = pd.DataFrame({"Actual": y_test, "prediction": y_pred})
print(comparison)

## let's plot the graph

plt.scatter(X_test, y_test, color = 'Red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('SALARY OF THE EMPLOYEES BASED ON EXPERIENCE')
plt.xlabel ('Experience')
plt.ylabel('Salary')
plt.show()

## To predict the future 
## as equation --- y= mx+c , we got m and c  now 

m_coef = regressor.coef_
print(m_coef)

c_intercept = regressor.intercept_
print(c_intercept)

y_12 = m_coef*12 + c_intercept
y_12

## To check the model accuracy

bias = regressor.score(X_train, y_train)
print(bias)

variance = regressor.score(X_test, y_test)
print(variance)

dataset.mean()

dataset['Salary'].mean().astype(int)

dataset.median()

dataset['Salary'].median()

dataset['Salary'].mode()

dataset.var()

dataset['Salary'].var()

dataset.std()

dataset['Salary'].std()


from scipy.stats import variation

variation(dataset.values)

variation(dataset['Salary'])

dataset.corr()

dataset['Salary'].corr(dataset['YearsExperience'])


dataset.skew()

dataset['Salary'].skew()


dataset.sem()   #  This gives standard error

dataset['Salary'].sem()  


### calculating the Z -score 

import scipy.stats as stats 

dataset.apply(stats.zscore)  

''' We calculate Z-scores to standardize the data so 
all variables are on the same scale and to help detect outliers'''

stats.zscore(dataset['Salary'])


## Let's calculate Sum of Square Regression

# SSR ( PREDICTED - AVERAGE)

y_mean = np.mean(y)
SSR = np.sum((y_pred- y_mean)**2)
print(SSR)


# SSE ( ACTUAL - PREDICTED)

y = y[0:6] ## this slicing depends on train_test_split
SSE = np.sum((y- y_pred)**2)
print(SSE)

# SST (ACTUAL - AVERAGE )

mean_total = np.mean(dataset.values)
SST = np.sum((dataset.values - mean_total)**2)
print(SST)


## R square 

r_square = 1 - SSR/SST
print(r_square)

import pickle 

## saving the trained model to disk

## we are converting python file to  binary file

filename  = 'Linear_Regression_Model.pkl'
with open (filename, 'wb') as file :
    pickle.dump(regressor, file)
    print('Model has pickled and saved as Linear_Regression_Model.pkl')