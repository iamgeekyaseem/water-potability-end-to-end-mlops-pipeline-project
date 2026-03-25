import pandas as pd
import numpy as np
import os

# train_data = pd.read_csv(r"./data/raw/train.csv")
# test_data = pd.read_csv(r"./data/raw/test.csv")
def load_data(file_path:str)->pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise Exception(f"Error loading data from {file_path}: {e}")

def fill_missing_values(dataframes:pd.DataFrame)->pd.DataFrame:
    try:
        for column in dataframes.columns:
            if dataframes[column].isnull().any():
                mean_value = dataframes[column].mean()
                dataframes[column].fillna(mean_value, inplace=True)
        return dataframes
    except Exception as e:
        raise Exception(f"Error filling missing values: {e}")
    
# train_preprocessed_data = fill_missing_values(train_data)
# test_preprocessed_data = fill_missing_values(test_data)

# data_path = os.path.join("data", "processed")
# os.makedirs(data_path, exist_ok=True) 
def create_directory(path:str, subfolder:str)-> str:
    try:
        data_path = os.path.join(path, subfolder)
        os.makedirs(data_path, exist_ok = True)
        return data_path
    except Exception as e:
        raise Exception(f"Error creating directory: {e}")


# train_preprocessed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
# test_preprocessed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)  
def save_preprocessed_data(data:pd.DataFrame, path:str, filename:str)->None:
    try:
        data.to_csv(os.path.join(path, filename), index = False)
    except Exception as e:
        raise Exception(f"Error saving preprocessed data to {os.path.join(path, filename)}: {e}")

def main():
    train_data_path = r"./data/raw/train.csv"
    test_data_path = r"./data/raw/test.csv"
    try:
        train_data = load_data(train_data_path)
        test_data = load_data(test_data_path)
        train_preprocessed_data = fill_missing_values(train_data)
        test_preprocessed_data = fill_missing_values(test_data)
        data_path = create_directory("data", "processed")
        save_preprocessed_data(train_preprocessed_data, data_path, "train_processed.csv")
        save_preprocessed_data(test_preprocessed_data, data_path, "test_processed.csv")
    except Exception as e:
        raise Exception(f"Error in main function: {e}")


if __name__ == "__main__":
    main()