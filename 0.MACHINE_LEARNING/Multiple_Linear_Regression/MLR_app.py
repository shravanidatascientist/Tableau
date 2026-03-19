import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

dataset = pd.read_csv(r"C:\Users\Shravani\Downloads\5th - mlr\5th - mlr\MLR\House_data.csv")

dataset = dataset.drop(["id","date"], axis=1)


x = dataset.iloc[:, dataset.columns != "price"]

y = dataset.iloc[:, 2]

# Splitting dataset into training and testing dataset

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=1/3, random_state=0
)

# Training the model

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred =  regressor.predict(x_test)

m = regressor.coef_
print(m)

c = regressor.intercept_
print(c)

import statsmodels.api as sm

x = sm.add_constant(x)

x_np = x.values  # convert DataFrame to NumPy
x_opt = x_np[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]

regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt = np.delete(x_opt, [3,5,6,7,8,9,12,14,15,16,17], axis=1)

regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
print(regressor_OLS.summary())

bias = regressor.score(x_train, y_train)
bias

variance = regressor.score(x_test, y_test)
variance

import pickle

filename = 'Linear_regression_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)
    print("model has been pickled and saved as Linear_regression_model.pkl")
    
    import streamlit as st

import streamlit as st

# Load the trained model
with open('Linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("House Price Prediction App")
st.write("Predict house prices based on 18 key features.")

# List of feature names (replace with your actual column names)
features_info = {
    "bedrooms": (1, 10, 3),
    "bathrooms": (1.0, 10.0, 2.0),
    "sqft_living": (300, 10000, 1500),
    "sqft_lot": (500, 50000, 5000),
    "floors": (1, 5, 1),
    "waterfront": (0, 1, 0),
    "view": (0, 4, 0),
    "condition": (1, 5, 3),
    "grade": (1, 13, 7),
    "sqft_above": (300, 8000, 1000),
    "sqft_basement": (0, 5000, 0),
    "yr_built": (1900, 2023, 1990),
    "yr_renovated": (0, 2023, 0),
    "zipcode": (98000, 98199, 98100),
    "lat": (47.0, 47.8, 47.5),
    "long": (-122.5, -121.5, -122.2),
    "sqft_living15": (300, 10000, 1500),
    "sqft_lot15": (500, 50000, 5000)
}

# Collect inputs using sliders
user_input = []
for feature, (min_val, max_val, default) in features_info.items():
    if isinstance(default, int):
        val = st.slider(f"{feature}", min_value=min_val, max_value=max_val, value=default)
    else:
        val = st.slider(f"{feature}", min_value=float(min_val), max_value=float(max_val), value=float(default))
    user_input.append(val)

# Predict button
if st.button("Predict Price"):
    input_array = np.array([user_input])  # shape (1,18)
    prediction = model.predict(input_array)
    st.success(f"Predicted House Price: ${prediction[0]:,.2f}")