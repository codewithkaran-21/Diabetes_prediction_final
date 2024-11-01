# app/models/input_model.py
from pydantic import BaseModel

class DiabetesPredictionInput(BaseModel):
    preg: int
    glucose: float
    bp: float
    skinthickness: float
    insulin: float
    bmi: float
    dpf: float
    age: int
