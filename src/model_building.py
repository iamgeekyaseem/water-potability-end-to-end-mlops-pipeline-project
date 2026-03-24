import pandas as pd
import numpy as np
import os

import pickle
from sklearn.ensemble import RandomForestClassifier

import yaml

train_data = pd.read_csv(r"./data/processed/train_processed.csv")

n_estimators = yaml.safe_load(open("params.yaml"))["model_building"]["n_estimators"]
random_state = yaml.safe_load(open("params.yaml"))["model_building"]["random_state"]
max_depth = yaml.safe_load(open("params.yaml"))["model_building"]["max_depth"]
min_samples_split = yaml.safe_load(open("params.yaml"))["model_building"]["min_samples_split"]
min_samples_leaf = yaml.safe_load(open("params.yaml"))["model_building"]["min_samples_leaf"]
max_features = yaml.safe_load(open("params.yaml"))["model_building"]["max_features"]
bootstrap = yaml.safe_load(open("params.yaml"))["model_building"]["bootstrap"]
n_jobs = yaml.safe_load(open("params.yaml"))["model_building"]["n_jobs"]


X_train = train_data.drop('Potability', axis=1)
y_train = train_data['Potability']

model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, bootstrap=bootstrap, n_jobs=n_jobs)
model.fit(X_train, y_train)

# Save the trained model
pickle.dump(model, open("model.pkl", "wb"))


