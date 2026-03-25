import pandas as pd
import numpy as np
import os

import pickle
from sklearn.ensemble import RandomForestClassifier

import yaml

def load_data(file_path:str)->pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise Exception(f"Error loading data from {file_path}: {e}")

def load_param(filepath: str, section:str, param_name:str):
    try:
        return yaml.safe_load(open(filepath))[section][param_name]
    except Exception as e:
        raise Exception(f"Error loading parameters from {filepath}: {e}")

def split_features_target(data:pd.DataFrame, target_column:str)-> tuple[pd.DataFrame, pd.Series]:
    try:
        X_train = data.drop(target_column, axis = 1)
        y_train = data[target_column]
        return X_train, y_train
    except Exception as e:
        raise Exception(f"Error splitting features and target: {e}")

def train_model(X_train:pd.DataFrame, y_train:pd.Series, n_estimators:int, random_state:int, max_depth:int, min_samples_split:int, min_samples_leaf:int, max_features, bootstrap:bool, n_jobs:int)-> RandomForestClassifier:
    try:
        model = RandomForestClassifier(n_estimators = n_estimators, 
                                       random_state = random_state, 
                                       max_depth = max_depth, 
                                       min_samples_split = min_samples_split, 
                                       min_samples_leaf = min_samples_leaf, 
                                       max_features = max_features, 
                                       bootstrap = bootstrap, 
                                       n_jobs = n_jobs)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        raise Exception(f"Error training model: {e}")

def save_model(model, filename:str)->None:
    try:
        pickle.dump(model, open(os.path.join(filename), "wb"))
    except Exception as e:
        raise Exception(f"Error saving model to {os.path.join(filename)}: {e}")

def main():
    train_data_path = r"./data/processed/train_processed.csv"
    param_path = r"params.yaml"
    stage_name = "model_building"
    try:
        train_data = load_data(train_data_path)
        n_estimators = load_param(param_path, stage_name, "n_estimators")
        random_state = load_param(param_path, stage_name, "random_state")
        max_depth = load_param(param_path, stage_name, "max_depth")
        min_samples_split = load_param(param_path, stage_name, "min_samples_split")
        min_samples_leaf = load_param(param_path, stage_name, "min_samples_leaf")
        max_features = load_param(param_path, stage_name, "max_features")
        bootstrap = load_param(param_path, stage_name, "bootstrap")
        n_jobs = load_param(param_path, stage_name, "n_jobs")
        X_train, y_train = split_features_target(train_data, target_column="Potability")
        model = train_model(X_train, y_train, n_estimators, random_state, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap, n_jobs)
        save_model(model, filename="model.pkl")
    except Exception as e:
        raise Exception(f"Error in main function: {e}")

if __name__ == "__main__":
    main()


