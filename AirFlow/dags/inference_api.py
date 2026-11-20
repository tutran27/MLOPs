from fastapi import FastAPI
from pydantic import BaseModel

from airflow.sdk import dag, task
import mlflow

app = FastAPI()

mlflow.set_tracking_uri("sqlite:///dags/mlflow.db")

RUN_ID = "c1b711862fe846c3a8647ebb9e90dc4d"
model = mlflow.sklearn.load_model(f"runs:/{RUN_ID}/model")

class PredictRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest):

    prediction = model.predict(
        [request.text]
    )[0]

    return {
        "prediction": int(prediction)
    }