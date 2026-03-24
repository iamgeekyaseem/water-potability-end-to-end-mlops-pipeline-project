import pandas as pd
import numpy as np

import pickle
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

test_data = pd.read_csv(r"./data/processed/test_processed.csv")

# X_test = test_data.iloc[:, :-1].values
# y_test = test_data.iloc[:, -1].values
# similar to the training data, we can directly use pandas DataFrames and Series for the test data as well, which allows us to keep the column names for better readability and debugging.

X_test = test_data.drop('Potability', axis=1)
y_test = test_data['Potability']

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

metrics_dict = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}

# Save metrics to a JSON file
with open("metrics.json", "w") as f:
    json.dump(metrics_dict, f, indent=4)