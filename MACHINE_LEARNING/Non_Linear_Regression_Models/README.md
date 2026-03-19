#  Machine Learning Regression Models

##  Project Overview
This project demonstrates the implementation of various **Machine Learning Regression algorithms**.  
It includes both **Linear** and **Non-Linear models**, along with performance comparison and ranking.

---

##  Models Implemented

###  Linear Models
- Simple Linear Regression  
- Multiple Linear Regression  
- Ridge Regression  
- Lasso Regression  

###  Non-Linear Models
- Polynomial Regression  
- Support Vector Regression (SVR)  
- K-Nearest Neighbors (KNN)  
- Decision Tree Regression  
- Random Forest Regression  

---

##  Model Evaluation

### Linear Models
- Evaluated individually using:
  - R² Score  
  - Train & Test Accuracy  

### Non-Linear Models
- Trained on the **same dataset**
- Compared using:
  - R² Score  
  - RMSE  

 A ranking table is created to identify the **Best Performing Model**

---

##  Results (Model Ranking)

| Rank | Model           | R² Score | RMSE |
|------|----------------|----------|------|
| 1    | Random Forest  | High     | Low  |
| 2    | Decision Tree  | High     | Low  |
| 3    | Polynomial     | Medium   | Medium |
| 4    | SVR            | Medium   | Medium |
| 5    | KNN            | Lower    | Higher |

---

##  Streamlit Dashboard

An interactive dashboard is built using Streamlit to:
- Display model performance  
- Show ranking table  
- Identify best model  

### Run Locally
```bash
streamlit run app.py

