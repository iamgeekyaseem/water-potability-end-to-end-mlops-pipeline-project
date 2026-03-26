import json
import pickle

# from dvclive import Live
import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow.data import from_pandas # pyright: ignore[reportAttributeAccessIssue]
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# import dagshub
# dagshub.init(repo_owner='iamgeekyaseem',
# repo_name='water-potability-end-to-end-mlops-pipeline-project',
# mlflow=True)


def load_data(file_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise Exception(f"Error loading data from {file_path}: {e}")


def split_features_target(
    data: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X_test = data.drop(target_column, axis=1)
        y_test = data[target_column]
        return X_test, y_test
    except Exception as e:
        raise Exception(f"Error splitting features and target: {e}")


def load_model(filename: str):
    try:
        model = pickle.load(open(filename, "rb"))
        return model
    except Exception as e:
        raise Exception(f"Error loading model from {filename}: {e}")


def make_predictions(model, X_test: pd.DataFrame):
    try:
        y_pred = model.predict(X_test)
        return y_pred
    except Exception as e:
        raise Exception(f"Error making predictions: {e}")


def calculate_metrics(y_test: pd.Series, y_pred: np.ndarray):
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
            "f1_score": f1,
        }
        with open(filename, "w") as f:
            json.dump(metrics_dict, f, indent=4)
    except Exception as e:
        raise Exception(f"Error saving metrics to {filename}: {e}")


def load_param(filepath: str, section: str, param_name: str):
    try:
        with open(filepath, "r") as file:
            params = yaml.safe_load(file)
        return params[section][param_name]
    except Exception as e:
        raise Exception(f"Error Loading parameters from {filepath}: {e}")


def experiment_tracking(metrics):
    try:
        # mlflow.set_tracking_uri("http://localhost:5000")
        # mlflow.set_tracking_uri("https://dagshub.com/iamgeekyaseem/water-potability-end-to-end-mlops-pipeline-project.mlflow")
        mlflow.set_experiment("model_evaluation")
        with mlflow.start_run():
            mlflow.log_metric("accuracy", metrics[0])
            mlflow.log_metric("precision", metrics[1])
            mlflow.log_metric("recall", metrics[2])
            mlflow.log_metric("f1_score", metrics[3])

            mlflow.log_artifact("metrics.json")
            mlflow.log_artifact("model.pkl")
            # mlflow.log_artifact("params.yaml")

            train_df = from_pandas(
                pd.read_csv("./data/processed/train_processed.csv"), name="train_data"
            )
            test_df = from_pandas(
                pd.read_csv("./data/processed/test_processed.csv"), name="test_data"
            )

            test_data = pd.read_csv("./data/processed/test_processed.csv")
            X_test = test_data.drop("Potability", axis=1)

            # logging dataset
            mlflow.log_input(train_df, context="training")
            mlflow.log_input(test_df, context="testing")

            mlflow.log_param("model_name", "RandomForestClassifier")
            mlflow.log_param(
                "n_estimators",
                load_param("params.yaml", "model_building", "n_estimators"),
            )
            mlflow.log_param(
                "random_state",
                load_param("params.yaml", "model_building", "random_state"),
            )
            mlflow.log_param(
                "max_depth", load_param("params.yaml", "model_building", "max_depth")
            )
            mlflow.log_param(
                "min_samples_split",
                load_param("params.yaml", "model_building", "min_samples_split"),
            )
            mlflow.log_param(
                "min_samples_leaf",
                load_param("params.yaml", "model_building", "min_samples_leaf"),
            )
            mlflow.log_param(
                "max_features",
                load_param("params.yaml", "model_building", "max_features"),
            )
            mlflow.log_param(
                "bootstrap", load_param("params.yaml", "model_building", "bootstrap")
            )
            mlflow.log_param(
                "n_jobs", load_param("params.yaml", "model_building", "n_jobs")
            )

            mlflow.log_artifact(__file__)
            model = load_model("model.pkl")
            sign = infer_signature(X_test, model.predict(X_test))
            mlflow.sklearn.log_model(model, "model", signature=sign) # pyright: ignore[reportPrivateImportUsage]

    except Exception as e:
        raise Exception(f"Error in experiment tracking: {e}")


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
        experiment_tracking(metrics)
    except Exception as e:
        raise Exception(f"Error in main function: {e}")


if __name__ == "__main__":
    main()
