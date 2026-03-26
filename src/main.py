import pickle

import pandas as pd
from fastapi import FastAPI

from data_model import Water

app = FastAPI(
    title="Water Potability Prediction API",
    description="An API to predict water potability based on various features.",
    version="1.0.0",
)


@app.get("/")
def index():
    return {"message": "Welcome to the Water Potability Prediction API!"}


with open(r"model.pkl", "rb") as f:
    model = pickle.load(f)


@app.post("/predict")
def predict(water: Water):
    input_data = pd.DataFrame(
        {
            "ph": [water.ph],
            "Hardness": [water.Hardness],
            "Solids": [water.Solids],
            "Chloramines": [water.Chloramines],
            "Sulfate": [water.Sulfate],
            "Conductivity": [water.Conductivity],
            "Organic_carbon": [water.Organic_carbon],
            "Trihalomethanes": [water.Trihalomethanes],
            "Turbidity": [water.Turbidity],
        }
    )
    prediction = model.predict(input_data)
    potability = "Potable" if prediction[0] == 1 else "Not Potable"

    return {"prediction": potability}
