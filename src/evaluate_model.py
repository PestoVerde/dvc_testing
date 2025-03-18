import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load test data
test = pd.read_csv("data/test.csv")

# Features and target
X_test = test.drop(columns=["price"])
y_test = test["price"]

# Load model
model = joblib.load("model/model.pkl")

# Predictions
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Save metrics
with open("metrics/metrics.txt", "w") as f:
    f.write(f"MAE: {mae}\nMSE: {mse}\n")