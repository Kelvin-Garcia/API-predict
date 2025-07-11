# predict_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
from fastapi.middleware.cors import CORSMiddleware

# Cargar modelo entrenado
model = joblib.load("random_forest_model.pkl")

# Instancia de la app
app = FastAPI(title="API de Predicción de Hipertensión")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes reemplazar "*" por dominios específicos si lo prefieres
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo de datos esperado
target_features = [
    "male", "age", "currentSmoker", "cigsPerDay", "BPMeds", "diabetes", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"
]

class InputData(BaseModel):
    male: int
    age: int
    currentSmoker: int
    cigsPerDay: float
    BPMeds: float
    diabetes: int
    totChol: float
    sysBP: float
    diaBP: float
    BMI: float
    heartRate: float
    glucose: float

@app.get("/")
def root():
    return {"mensaje": "API de predicción de hipertensión lista"}

@app.post("/predict")
def predict(data: InputData):
    try:
        # Convertir entrada a DataFrame
        input_df = pd.DataFrame([data.dict()])
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]  # Probabilidad de clase positiva
        proba_percent = round(float(proba) * 100, 2)

        if prediction == 0:
            resultado = "Sin riesgo de Hipertensión"
        else:
            resultado = "Riesgo de Hipertensión"

        return {
            "mensaje": resultado,
            "probabilidad de hipertension": f"{proba_percent} %"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("predict_api:app", host="0.0.0.0", port=port, reload=False)
