# simple_ml_model.py

import numpy as np
from sklearn.linear_model import LinearRegression
import mlflow
import mlflow.sklearn

# Sample data
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3

# Train a model
model = LinearRegression()
model.fit(X, y)

# Log the model
with mlflow.start_run():
    mlflow.sklearn.log_model(model, "linear_regression_model")
