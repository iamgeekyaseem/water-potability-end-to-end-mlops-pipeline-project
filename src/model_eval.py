import pandas as pd
import numpy as np

import pickle
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# test_data = pd.read_csv(r"./data/processed/test_processed.csv")
def load_data(file_path:str)->pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise Exception(f"Error loading data from {file_path}: {e}")

# X_test = test_data.iloc[:, :-1].values
# y_test = test_data.iloc[:, -1].values
# similar to the training data, we can directly use pandas DataFrames and Series for the test data as well, which allows us to keep the column names for better readability and debugging.

# X_test = test_data.drop('Potability', axis=1)
# y_test = test_data['Potability']
def split_features_target(data:pd.DataFrame, target_column:str)-> tuple[pd.DataFrame, pd.Series]:
    try:
        X_test = data.drop(target_column, axis = 1)
        y_test = data[target_column]
        return X_test, y_test
    except Exception as e:
        raise Exception(f"Error splitting features and target: {e}")

# Load the trained model
# model = pickle.load(open("model.pkl", "rb"))
def load_model(filename:str):
    try:
        model = pickle.load(open(filename, "rb"))
        return model
    except Exception as e:
        raise Exception(f"Error loading model from {filename}: {e}")

# Make predictions on the test data
# y_pred = model.predict(X_test)
def make_predictions(model, X_test:pd.DataFrame)-> np.ndarray:
    try:
        y_pred = model.predict(X_test)
        return y_pred
    except Exception as e:
        raise Exception(f"Error making predictions: {e}")

# Calculate evaluation metrics
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
def calculate_metrics(y_test:pd.Series, y_pred:np.ndarray):
    try:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        return accuracy, precision, recall, f1
    except Exception as e:
        raise Exception(f"Error calculating metrics: {e}")



# metrics_dict = {
#     "accuracy": accuracy,
#     "precision": precision,
#     "recall": recall,
#     "f1_score": f1
# }
# # Save metrics to a JSON file
# with open("metrics.json", "w") as f:
#     json.dump(metrics_dict, f, indent=4)
def save_metrics(metrics, filename: str):
    try:
        accuracy, precision, recall, f1 = metrics
        metrics_dict = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        with open(filename, "w") as f:
            json.dump(metrics_dict, f, indent=4)
    except Exception as e:
        raise Exception(f"Error saving metrics to {filename}: {e}")

def main():
    test_data_path = r"./data/processed/test_processed.csv"
    model_path = r"model.pkl"
    metrics_path = r"metrics.json"
    try:
        test_data = load_data(test_data_path)
        X_test, y_test = split_features_target(test_data, target_column="Potability")
        model = load_model(model_path)
        y_pred = make_predictions(model, X_test)
        metrics = calculate_metrics(y_test, y_pred)
        save_metrics(metrics, metrics_path)
    except Exception as e:
        raise Exception(f"Error in main function: {e}")

if __name__ == "__main__":
    main()

