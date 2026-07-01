import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn

# ==========================================
# 1. Cargar el modelo entrenado
# ==========================================


modelo = joblib.load('modelo_arbol.pkl')

# ==========================================
# 2. Definir la app FastAPI
# ==========================================

app = FastAPI(
    title="FitPredict - API de scoring crediticio",
    description="API para predecir si un cliente pagara su prestamo a tiempo",
    version="1.0.0"
)

# ==========================================
# 3. Definir el esquema de entrada con Pydantic
# ==========================================

class Solicitud(BaseModel):
    tipo_credito: int
    fecha_prestamo: int
    capital_prestado: int
    plazo_meses: int
    edad_cliente: int
    tipo_laboral: str
    salario_cliente: int
    total_otros_prestamos: int
    cuota_pactada: int
    puntaje: str
    puntaje_datacredito: float
    cant_creditosvigentes: int
    huella_consulta: int
    saldo_mora: float
    saldo_total: float
    saldo_principal: float
    saldo_mora_codeudor: float
    creditos_sectorFinanciero: int
    creditos_sectorCooperativo: int
    creditos_sectorReal: int
    promedio_ingresos_datacredito: float
    tendencia_ingresos: str

class BatchSolicitud(BaseModel):
    solicitudes: List[Solicitud]

# ==========================================
# 4. Endpoints
# ==========================================

@app.get("/")
def root():
    return {"mensaje": "API de scoring crediticio FitPredict activa"}

@app.post("/predict")
def predict(batch: BatchSolicitud):
    datos = pd.DataFrame([s.dict() for s in batch.solicitudes])
    predicciones = modelo.predict(datos)
    probabilidades = modelo.predict_proba(datos)[:, 1]

    resultados = []
    for i, (pred, prob) in enumerate(zip(predicciones, probabilidades)):
        resultados.append({
            "solicitud": i + 1,
            "prediccion": int(pred),
            "resultado": "Paga a tiempo" if pred == 1 else "No paga a tiempo",
            "probabilidad": round(float(prob), 4)
        })

    return {"resultados": resultados}

# ==========================================
# 5. Correr la app
# ==========================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)