import os

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_param(filepath: str, section: str, param_name: str):
    try:
        with open(filepath, "r") as file:
            params = yaml.safe_load(file)
        return params[section][param_name]
    except Exception as e:
        raise Exception(f"Error Loading parameters from {filepath}: {e}")


def load_data(filepath: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(filepath)
        return data
    except Exception as e:
        raise Exception(f"Error Loading data from {filepath}: {e}")


def split_data(
    data: pd.DataFrame, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        train_data, test_data = train_test_split(
            data, test_size=test_size, random_state=random_state
        )
        return train_data, test_data
    except Exception as e:
        raise Exception(f"Error splitting data: {e}")


def create_data_path(base_path: str, subfolder: str) -> str:
    try:
        data_path = os.path.join(base_path, subfolder)
        return data_path
    except Exception as e:
        raise Exception(f"Error creating data path: {e}")


def create_directory(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        raise Exception(f"Error creating directory: {e}")


def save_data(data: pd.DataFrame, path: str, filename: str) -> None:
    try:
        data.to_csv(os.path.join(path, filename), index=False)
    except Exception as e:
        raise Exception(f"Error saving data to {os.path.join(path, filename)}: {e}")


def main():
    dataset_path = r"/Users/aseem/projects/mlpipeline/water_potability.csv"
    param_path = r"params.yaml"
    stage_name = "data_collection"
    try:
        data = load_data(dataset_path)
        test_size = load_param(param_path, stage_name, "test_size")
        random_state = load_param(param_path, stage_name, "random_state")
        train_data, test_data = split_data(
            data, test_size=test_size, random_state=random_state
        )
        data_path = create_data_path("data", "raw")
        create_directory(data_path)
        save_data(train_data, data_path, "train.csv")
        save_data(test_data, data_path, "test.csv")
    except Exception as e:
        raise Exception(f"Error in main function: {e}")


if __name__ == "__main__":
    main()
