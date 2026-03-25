import pandas as pd
import numpy as np

import pickle
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_data(file_path:str)->pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise Exception(f"Error loading data from {file_path}: {e}")

def split_features_target(data:pd.DataFrame, target_column:str)-> tuple[pd.DataFrame, pd.Series]:
    try:
        X_test = data.drop(target_column, axis = 1)
        y_test = data[target_column]
        return X_test, y_test
    except Exception as e:
        raise Exception(f"Error splitting features and target: {e}")

def load_model(filename:str):
    try:
        model = pickle.load(open(filename, "rb"))
        return model
    except Exception as e:
        raise Exception(f"Error loading model from {filename}: {e}")

def make_predictions(model, X_test:pd.DataFrame):
    try:
        y_pred = model.predict(X_test)
        return y_pred
    except Exception as e:
        raise Exception(f"Error making predictions: {e}")

def calculate_metrics(y_test:pd.Series, y_pred:np.ndarray):
    try:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        return accuracy, precision, recall, f1
    except Exception as e:
        raise Exception(f"Error calculating metrics: {e}")

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

