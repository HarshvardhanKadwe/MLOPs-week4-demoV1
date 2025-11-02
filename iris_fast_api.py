from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Initialize FastAPI app
# ------------------------------------------------------
app = FastAPI(
    title="Iris Prediction API",
    description="A FastAPI service that predicts Iris flower species",
    version="1.0"
)


# ------------------------------------------------------
# Load the ML model
# ------------------------------------------------------
model = joblib.load("model.pkl")

# ------------------------------------------------------
# Define input schema for Iris dataset
# ------------------------------------------------------
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Iris class labels
iris_labels = ["setosa", "versicolor", "virginica"]

# ------------------------------------------------------
@app.get("/")
def read_root():
    return {"message" : "Welcome to Iris Data Classifier"}
# ------------------------------------------------------
# Prediction Endpoint
# ------------------------------------------------------
@app.post("/predict")
def predict_iris(data: IrisInput):
    # Convert input into dataframe
    
    input_df = pd.DataFrame([data.dict()])
    predictions = model.predict(input_df)[0]
    
    return {
        "predicted_class" : predictions
    }