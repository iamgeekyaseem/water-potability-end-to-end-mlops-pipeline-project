import pandas as pd
import numpy as np
import os

import pickle
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv(r"./data/processed/train_processed.csv")

# X_train = train_data.iloc[:, :-1].values
# y_train = train_data.iloc[:, -1].values
#becasue numpy arrays are not needed for scikit-learn models, we can directly use pandas DataFrames and Series for training the model. This allows us to keep the column names, which can be helpful for debugging and understanding the model's behavior.

X_train = train_data.drop('Potability', axis=1)
y_train = train_data['Potability']

model = RandomForestClassifier(n_estimators=100, random_state=4)
model.fit(X_train, y_train)

# Save the trained model
pickle.dump(model, open("model.pkl", "wb"))


