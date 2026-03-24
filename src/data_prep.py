import pandas as pd
import numpy as np
import os

train_data = pd.read_csv(r"./data/raw/train.csv")
test_data = pd.read_csv(r"./data/raw/test.csv")

def fill_missing_values(dataframes):
    for column in dataframes.columns:
        if dataframes[column].isnull().any():
            mean_value = dataframes[column].mean()
            dataframes[column].fillna(mean_value, inplace=True)
    return dataframes

train_preprocessed_data = fill_missing_values(train_data)
test_preprocessed_data = fill_missing_values(test_data)

data_path = os.path.join("data", "processed")
os.makedirs(data_path, exist_ok=True) 

train_preprocessed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
test_preprocessed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)  