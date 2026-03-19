import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

dataset = pd.read_csv(r"C:\Users\shravani\Desktop\NIT\emp_sal.csv")

x = dataset.iloc[:, 1:2]

y = dataset.iloc [:,-1]

from sklearn.svm import SVR

svr_reg = SVR()
svr_reg.fit(x,y)

## let's predict 

svr_pred = svr_reg.predict([[6.5]])
print(svr_pred)

## Hyper paramter tunning

svr_reg = SVR(kernel = 'poly', degree = 4, gamma ='auto')
svr_reg.fit(x,y)

svr_pred = svr_reg.predict([[6.5]])
print(svr_pred)


svr_reg = SVR(kernel = 'poly', degree = 5, gamma ='scale')
svr_reg.fit(x,y)

svr_pred = svr_reg.predict([[6.5]])
print(svr_pred)