import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

dataset = pd.read_csv(r"C:\Users\shravani\Desktop\NIT\emp_sal.csv")

x = dataset.iloc[:, 1:2]

y = dataset.iloc [:,-1]

from sklearn.neighbors import KNeighborsRegressor

knn_reg = KNeighborsRegressor()
knn_reg.fit(x,y)

## let's predict 

knn_pred = knn_reg.predict([[6.5]])
print(knn_pred)


## Lets do hyperparamter tunning

knn_reg = KNeighborsRegressor(n_neighbors=2)
knn_reg.fit(x,y)

knn_pred = knn_reg.predict([[6.5]])
print(knn_pred)