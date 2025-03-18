import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# Load train data
train = pd.read_csv("data/train.csv")

# Features and target
X_train = train.drop(columns=["price"])
y_train = train["price"]

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model/model.pkl")