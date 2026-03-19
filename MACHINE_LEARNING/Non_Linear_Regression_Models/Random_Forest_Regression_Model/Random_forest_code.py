import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

dataset = pd.read_csv(r"C:\Users\shravani\Desktop\NIT\emp_sal.csv")

x = dataset.iloc[:, 1:2]

y = dataset.iloc [:,-1]


from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor()
rf_reg.fit(x,y)

## Lets predict

rf_pred = rf_reg.predict([[6.5]])
print(rf_pred)


## Let's do hyperparameter tunning

rf_reg = RandomForestRegressor(n_estimators = 200, random_state=0,max_depth=10, min_samples_split=5, min_samples_leaf=2)
rf_reg.fit(x,y)


rf_pred = rf_reg.predict([[6.5]])
print(rf_pred)