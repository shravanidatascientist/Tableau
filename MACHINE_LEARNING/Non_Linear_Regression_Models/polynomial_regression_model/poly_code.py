import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

dataset = pd.read_csv(r"C:\Users\shravani\Desktop\NIT\emp_sal.csv")

x = dataset.iloc[:, 1:2]

y = dataset.iloc [:,-1]

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(x,y)


lin_model_pred = lin_reg.predict([[6.5]])
lin_model_pred


plt.scatter(x,y ,color = 'red')
plt.plot(x, lin_reg.predict(x),color ='blue')
plt.title('Linear Regression Graph')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

## POLYNOMIAL FEATURES ---- BY DEFAULT DEGREE 2 

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree= 5)
x_poly = poly_reg.fit_transform(x)

poly_reg.fit(x_poly,y)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)

plt.scatter(x,y, color = 'red')
plt.plot(x, lin_reg2.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.title(' Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


poly_model_pred = lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
poly_model_pred