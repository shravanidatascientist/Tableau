import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

# Loading the dataset

dataset = pd.read_csv(r"C:\Users\shravani\Desktop\NIT\Investment.csv")

print(dataset)

## separating independent variables and dependent varaibles 

x = dataset.iloc[:,:-1]
y = dataset.iloc[:, 4]

'''   X → Everything except output---- y → Only output '''

## As we have catgorical data in the dataset, we will convertcategorical values to numerical values by using get_dummies()

x = pd.get_dummies(x, dtype=int)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

## from the equation constant is missing, let's find the constant 

interept = regressor.intercept_
print(interept)

coef = regressor.coef_
print(coef)

''' up to this we build multiple linear regresssion model'''

# Adding constant to the dataset

x = np.append(arr = np.full((50,1), 42467).astype(int), values = x, axis =1 )

'''we use this np.append() in multiplelinear refression for backward elimination'''

## api is a connector which connects backend to fronted and frontend to backend
## Feature Elimination or Feature Selection ---- 


import statsmodels.api as sm

x_opt = x[:, [0,1,2,3,4,5]]

## ORDINARY LEAST SQUARE 

regressor_OLS = sm.OLS(endog =y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0,1,2,3,5]]
regressor_OLS = sm.OLS(endog =y, exog = x_opt).fit()
regressor_OLS.summary()


x_opt = x[:, [0,1,2,3]]
regressor_OLS = sm.OLS(endog =y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:, [0,1]]
regressor_OLS = sm.OLS(endog =y, exog = x_opt).fit()
regressor_OLS.summary()

bias = regressor.score(x_train, y_train)
print(bias)

variance = regressor.score(x_test, y_test)
print(variance)