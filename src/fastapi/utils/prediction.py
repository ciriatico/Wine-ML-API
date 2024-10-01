import numpy as np
from models.prediction import Prediction

def parse_prediction(prediction_model: np.ndarray) -> Prediction:
    prediction_label, prediction_score = str(prediction_model.argmax()), float(prediction_model.max())

    prediction_label = f"class_{prediction_label}"
    prediction_score = round(prediction_score, 5)

    return Prediction(prediction_label=prediction_label, prediction_score=prediction_score)
