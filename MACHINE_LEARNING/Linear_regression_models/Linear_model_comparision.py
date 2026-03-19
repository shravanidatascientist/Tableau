import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn import preprocessing

# Load dataset
dataset = pd.read_csv(r"C:\Users\shravani\Desktop\NIT\car-mpg.csv")

# Data cleaning
dataset = dataset.drop(columns=['car_name'])
dataset['origin'] = dataset['origin'].replace({1:'america',2:'europe',3:'asia'})
dataset = pd.get_dummies(dataset, columns=['origin'], dtype=int)

dataset = dataset.replace('?', np.nan)
dataset = dataset.apply(pd.to_numeric, errors='ignore')
dataset = dataset.apply(lambda x: x.fillna(x.median()) if x.dtype != 'object' else x)

# Split X and y
X = dataset.drop(['mpg'], axis=1)
y = dataset['mpg']

# Scaling
X = preprocessing.scale(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

results = []

# -----------------------------
# 1. Linear Regression
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

results.append({
    "Model": "Linear Regression",
    "R2": r2_score(y_test, y_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
})

# -----------------------------
# 2. Ridge
# -----------------------------
model = Ridge(alpha=0.3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

results.append({
    "Model": "Ridge",
    "R2": r2_score(y_test, y_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
})

# -----------------------------
# 3. Lasso
# -----------------------------
model = Lasso(alpha=0.1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

results.append({
    "Model": "Lasso",
    "R2": r2_score(y_test, y_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
})

# Create DataFrame
df_linear = pd.DataFrame(results)

# Ranking
df_linear["Rank"] = df_linear["R2"].rank(ascending=False)

df_linear = df_linear.sort_values("Rank")

print(df_linear)