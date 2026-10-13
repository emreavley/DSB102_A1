"""
TLDR:
Best λ: 17.4753
Test RMSE (Ridge): 31.1198

Coefficients 
-------------------------
Price: -19.7
Display: 18.11
Media: 12.09
"""

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SequentialFeatureSelector
import warnings; warnings.filterwarnings('ignore')

df = pd.read_csv("retail_dynamics_weekly_sales.csv")
y = df[["Week", "Sales"]]
df = df.drop(columns="StoreID")

# print(df.head())

# Next is to make a function to standardise "Display", I want to convert a No to 0 and a yes to 1
def StandardisingDisplay(row):
    stringToBeConverted = row["Display"]
    if stringToBeConverted[0] == "N" or stringToBeConverted[0] == "n":
        return 0
    else:
        return 1

df["Display"] = df.apply(StandardisingDisplay, axis=1)
X = df[["Week", "Price", "Display", "Media"]]
X_train = X[X["Week"] <= 8]
y_train = y[y["Week"] <= 8]
X_test = X[X["Week"] > 8]
y_test = y[y["Week"] > 8]

X_train = X_train.drop(columns="Week")
X_test = X_test.drop(columns="Week")
y_train = y_train.drop(columns="Week")
y_test = y_test.drop(columns="Week")
X = X.drop(columns="Week")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

alphas = np.logspace(-3, 4, 100)
coefs  = np.array([Ridge(alpha=a).fit(X_train, y_train).coef_ for a in alphas])

ridge_cv = RidgeCV(alphas=alphas, cv=5).fit(X_train, y_train)
print(f"Best λ: {ridge_cv.alpha_:.4f}")
print(f"Test RMSE (Ridge): {np.sqrt(mean_squared_error(y_test, ridge_cv.predict(X_test))):.4f}")
print(f"Coefficients")
print("-------------------------")
for i in range(0, 3):
    print(f"{X.columns[i]}: {round(ridge_cv.coef_[i], 2)}")