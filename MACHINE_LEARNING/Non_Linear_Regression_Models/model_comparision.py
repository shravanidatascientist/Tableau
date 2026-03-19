import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Load dataset
dataset = pd.read_csv(r"C:\Users\shravani\Desktop\NIT\emp_sal.csv")

X = dataset.iloc[:, 1:2]
y = dataset.iloc[:, -1]

# Train-Test Split (IMPORTANT)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

results = []

# -----------------------------
# 1. Polynomial Regression
# -----------------------------
poly = PolynomialFeatures(degree=5)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

y_pred = model.predict(X_test_poly)

results.append({
    "Model": "Polynomial",
    "R2": r2_score(y_test, y_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
})

# -----------------------------
# 2. SVR
# -----------------------------
model = SVR(kernel='poly', degree=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

results.append({
    "Model": "SVR",
    "R2": r2_score(y_test, y_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
})

# -----------------------------
# 3. KNN
# -----------------------------
model = KNeighborsRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

results.append({
    "Model": "KNN",
    "R2": r2_score(y_test, y_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
})

# -----------------------------
# 4. Decision Tree
# -----------------------------
model = DecisionTreeRegressor(random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

results.append({
    "Model": "Decision Tree",
    "R2": r2_score(y_test, y_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
})

# -----------------------------
# 5. Random Forest
# -----------------------------
model = RandomForestRegressor(n_estimators=200, random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

results.append({
    "Model": "Random Forest",
    "R2": r2_score(y_test, y_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
})

# -----------------------------
# Create Ranking Table
# -----------------------------
df = pd.DataFrame(results)

df["Rank"] = df["R2"].rank(ascending=False)

df = df.sort_values("Rank")

print(df)