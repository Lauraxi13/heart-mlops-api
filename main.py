# API FastAPI que recibe los datos clínicos de un paciente y consulta el endpoint
# heart-mlops-endpoint de Databricks (modelo workspace.heart_mlops.heart_disease_model@champion)
import os
from typing import Literal, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

# Credenciales: se inyectan como variables de entorno en Azure (nunca escritas en el código)
DATABRICKS_URL = os.getenv("DATABRICKS_URL")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

app = FastAPI(
    title="API Heart MLOps",
    description="Predicción de riesgo de enfermedad cardiaca (dataset heart_disease_uci)",
    version="1.0.0",
)

# Permite que un frontend (index.html) llame a la API desde otro dominio
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# Las 11 variables de la firma registrada en MLflow (NB02), ni una más ni una menos.
# extra="forbid" rechaza con 422 cualquier campo que no sea del modelo.
# Todas las claves son obligatorias. Las que el dataset trae con faltantes aceptan null:
# el Pipeline del modelo las imputa (mediana, moda o "desconocido").
class Paciente(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {
            "age": 67, "trestbps": 160, "chol": 286, "thalch": 108, "oldpeak": 1.5,
            "fbs": 0, "exang": 1, "sex": "Male", "cp": "asymptomatic",
            "restecg": "lv hypertrophy", "slope": "flat",
        }},
    )

    # Numéricas (mismos rangos que las reglas de calidad de la capa silver, NB01)
    age: float = Field(..., ge=18, le=100, description="Edad en años")
    trestbps: Optional[float] = Field(..., ge=80, le=220, description="Presión arterial en reposo (mm Hg)")
    chol: Optional[float] = Field(..., ge=50, le=700, description="Colesterol sérico (mg/dl)")
    thalch: Optional[float] = Field(..., ge=60, le=220, description="Frecuencia cardiaca máxima alcanzada")
    oldpeak: Optional[float] = Field(..., ge=-3, le=7, description="Depresión del ST inducida por ejercicio")

    # Binarias (0 = no, 1 = sí)
    fbs: Optional[Literal[0, 1]] = Field(..., description="Glucemia en ayunas > 120 mg/dl")
    exang: Optional[Literal[0, 1]] = Field(..., description="Angina inducida por ejercicio")

    # Categóricas (texto exacto, como en el dataset)
    sex: Literal["Male", "Female"]
    cp: Literal["typical angina", "atypical angina", "non-anginal", "asymptomatic"]
    restecg: Optional[Literal["normal", "st-t abnormality", "lv hypertrophy"]] = Field(...)
    slope: Optional[Literal["upsloping", "flat", "downsloping"]] = Field(...)


def a_registro_modelo(p: Paciente) -> dict:
    """Convierte al paciente en el registro que exige la firma: numéricas y binarias como double."""
    reg = p.model_dump()
    for col in ["age", "trestbps", "chol", "thalch", "oldpeak", "fbs", "exang"]:
        if reg[col] is not None:
            reg[col] = float(reg[col])
    return reg


@app.get("/health")
def health():
    # Verifica que la API está viva y que tiene configuradas las credenciales
    return {"status": "ok", "databricks_configurado": bool(DATABRICKS_URL and DATABRICKS_TOKEN)}


@app.post("/predecir")
def predecir(paciente: Paciente):
    if not (DATABRICKS_URL and DATABRICKS_TOKEN):
        raise HTTPException(status_code=500, detail="Faltan DATABRICKS_URL o DATABRICKS_TOKEN")

    # Databricks exige el formato dataframe_records
    payload = {"dataframe_records": [a_registro_modelo(paciente)]}

    try:
        r = requests.post(
            DATABRICKS_URL,
            headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"},
            json=payload,
            timeout=120,  # margen para el arranque en frío del endpoint (scale to zero)
        )
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Databricks tardó demasiado (arranque en frío). Intente de nuevo.")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"No se pudo conectar con Databricks: {e.__class__.__name__}")

    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Error de Databricks: {r.text}")

    pred = int(r.json()["predictions"][0])  # 1 = enfermedad cardiaca, 0 = sin enfermedad
    # "predictions" conserva el formato del endpoint de Databricks; los otros campos lo explican
    return {
        "predictions": [pred],
        "resultado": "Riesgo alto de enfermedad cardiaca" if pred == 1 else "Riesgo bajo de enfermedad cardiaca",
        "aviso": "Resultado de un modelo académico; no reemplaza un diagnóstico médico.",
    }
