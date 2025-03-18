import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/diamonds.csv")

# Select relevant features and encode categorical variables
# df = df.drop(columns=["Unnamed: 0"])
df = pd.get_dummies(df, columns=["cut", "color", "clarity"], drop_first=True)

# Split into train and test sets
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Save datasets
train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)