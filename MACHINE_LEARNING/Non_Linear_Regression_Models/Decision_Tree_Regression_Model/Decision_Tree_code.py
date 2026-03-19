import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

dataset = pd.read_csv(r"C:\Users\shravani\Desktop\NIT\emp_sal.csv")

x = dataset.iloc[:, 1:2]

y = dataset.iloc [:,-1]

## Decision Tree

from sklearn.tree import DecisionTreeRegressor

dt_reg = DecisionTreeRegressor()
dt_reg.fit(x,y)

## Let's predict 

dt_pred = dt_reg.predict([[6.5]])
print(dt_pred)

## Let's do hyperparameter tunning

dt_reg = DecisionTreeRegressor( criterion="absolute_error",random_state = 0)
dt_reg.fit(x,y)

dt_pred = dt_reg.predict([[6.5]])
print(dt_pred)
