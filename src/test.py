import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(BASE_DIR, "data/raw/water_potability.csv")

print(BASE_DIR)

print(dataset_path)