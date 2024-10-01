from fastapi import APIRouter
from xgboost import XGBClassifier
import numpy as np
from models.prediction import Feature, Prediction
from utils.prediction import parse_prediction
from utils.ml_models import get_path_last_version

router = APIRouter(prefix="/api")

model_path = get_path_last_version("wine_classification_xgboost_gridsearch")
model = XGBClassifier()
model.load_model(model_path)

@router.get("/health")
async def root():
    return {"message": "Estou saudavel"}

@router.post("/predict")
async def predict(features: Feature) -> Prediction:
    feature_values = [list(features.dict().values())]

    prediction_model = model.predict_proba(feature_values)[0]
    prediction = parse_prediction(prediction_model)

    return prediction
