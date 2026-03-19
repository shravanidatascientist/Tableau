
''' Regularization techniques are used to prevent overfitting by 

adding a penalty termto the regression cost function. The three most common ones are 

Ridge, Lasso, and Elastic Net (often used with libraries like scikit-learn and NumPy)
--------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
1) Ridge → Square penalty -- Regularize from high coefficient to low coefficients 
to reduce overfitting

2) Lasso → Absolute penalty--- it is alos called as feacture elimination or
 feature selection technique we regularize high coefficients to 0 , so that  reduces attributes 

3) Elastic Net → Both '''



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns


from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import r2_score

dataset = pd.read_csv(r"C:\Users\shravani\Desktop\NIT\car-mpg.csv")

#Drop car name
#Replace origin into 1,2,3.. dont forget get_dummies
#Replace ? with nan
#Replace all nan with median


dataset = dataset.drop(columns=['car_name'], axis =1)

dataset['origin'] = dataset['origin'].replace({1:'america',2:'europe',3:'asia'})

dataset = pd.get_dummies(dataset, columns=['origin'], dtype = int)

dataset = dataset.replace('?', np.nan)

dataset = dataset.apply(pd.to_numeric, errors='ignore')

dataset = dataset.apply(lambda x:x.fillna(x.median()) if x.dtype!= 'object' else x)

''' First we divide the data into independent (X) and dependent data (y) then we scale it. 

#### Tip!: ####

The reason we don't scale the entire data before and then divide it into 
train(X) & test(y) is because once you scale the data, the type(data_s) would be numpy.ndarray. 
It's impossible to divide this data when it's an array. 


Hence we divide type(data) pandas.DataFrame, then proceed to scaling it'''


X = dataset.drop(['mpg'], axis = 1) # independent variable
y = dataset[['mpg']] #dependent variable

#Scaling the data

## Scaling = Making all features speak the same numerical language.

X_s = preprocessing.scale(X)
X_s = pd.DataFrame(X_s, columns = X.columns) #converting scaled data into dataframe

y_s = preprocessing.scale(y)
y_s = pd.DataFrame(y_s, columns = y.columns) #ideally train, test data should be in columns

#Split into train, test set

X_train, X_test, y_train,y_test = train_test_split(X_s, y_s, test_size = 0.30, random_state = 1)
X_train.shape


#Fit simple linear model and find coefficients
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

for idx, col_name in enumerate(X_train.columns):
    print('The coefficient for {} is {}'.format(col_name, regression_model.coef_[0][idx]))
    
intercept = regression_model.intercept_[0]
print('The intercept is {}'.format(intercept))

#alpha factor here is lambda (penalty term) which helps to reduce the magnitude of coeff

ridge_model = Ridge(alpha = 0.3)
ridge_model.fit(X_train, y_train)

print('Ridge model coef: {}'.format(ridge_model.coef_))
#As the data has 10 columns hence 10 coefficients appear here   

#alpha factor here is lambda (penalty term) which helps to reduce the magnitude of coeff

lasso_model = Lasso(alpha = 0.1)
lasso_model.fit(X_train, y_train)

print('Lasso model coef: {}'.format(lasso_model.coef_))
#As the data has 10 columns hence 10 coefficients appear here   


#Model score - r^2 or coeff of determinant
#r^2 = 1-(RSS/TSS) = Regression error/TSS 


#Simple Linear Model
print(regression_model.score(X_train, y_train))
print(regression_model.score(X_test, y_test))

print('*************************')
#Ridge
print(ridge_model.score(X_train, y_train))
print(ridge_model.score(X_test, y_test))

print('*************************')
#Lasso
print(lasso_model.score(X_train, y_train))
print(lasso_model.score(X_test, y_test)) 