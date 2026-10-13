# -*- coding: utf-8 -*-
Original file is located at
    https://colab.research.google.com/drive/17rJFs68jZJH_QAeJLA0ZNSGce1Q2GoTG
"""

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the data
data = pd.read_csv("retail_dynamics_weekly_sales.csv")
# Clean the data
data = data.drop(columns="StoreID")
# Standardise display to convert no/yes - REWRITE
def StandardisingDisplay(row):
    stringToBeConverted = row["Display"]
    if stringToBeConverted[0] == "N" or stringToBeConverted[0] == "n":
        return 0
    else:
        return 1

# Apply the standardisation to the DataFrame
data["Display"] = data.apply(StandardisingDisplay, axis=1)

# Allocate features to X and select the target (sales)
X = data[["Week", "Price", "Display", "Media"]]
Y = data["Sales"]

# Split the training and test data up
X_train = X.iloc[:8]
X_test = X.iloc[8:]

Y_train = Y.iloc[:8]
Y_test = Y.iloc[8:]

# Create the model
model = RandomForestRegressor(n_estimators = 100, max_depth = 3, random_state = 42)

# Train the model on the first 8 weeks
model.fit(X_train, Y_train)

# Predict the sales for weeks 9 and 10
predictions = model.predict(X_test)

print("Actual sales:", Y_test.tolist()[:10], "... (truncated for display)") # convert data into a list
print("Predicted sales:", predictions.round(2)[:10], "... (truncated for display)") # round predicted sales

# Calculate evaluation metrics
MAE = mean_absolute_error(Y_test, predictions)
RSME = np.sqrt(mean_squared_error(Y_test, predictions))

print(f"Mean Absolute Error (MAE): {MAE:.2f}")
print(f"Root Mean Squared Error (RMSE): {RSME:.2f}")
