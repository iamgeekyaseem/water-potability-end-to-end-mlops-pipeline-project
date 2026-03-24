import pandas as pd
import numpy as np
import os

import pickle
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv(r"./data/processed/train_processed.csv")

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values

model = RandomForestClassifier(n_estimators=100, random_state=4)
model.fit(X_train, y_train)

# Save the trained model
pickle.dump(model, open("model.pkl", "wb"))



