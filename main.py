from pathlib import Path
import math
import statistics
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo_alertastock_regresion.pkl"

app = FastAPI(
    title="AlertaStock IA API",
    version="1.0.0",
    description="Prediccion de demanda y reglas operativas de riesgo/reposicion para AlertaStock.",
)

modelo = joblib.load(MODEL_PATH)


class PrediccionRequest(BaseModel):
    lag_1: float
    lag_2: float
    lag_3: float
    lag_4: float
    promedio_movil_4: float
    promedio_movil_8: float
    tendencia_4: float
    descuento_pct: float = Field(default=0.0, ge=0.0)
    mes: int = Field(ge=1, le=12)
    semana_del_ano: int = Field(ge=1, le=53)
    categoria: str
    stockActual: int = Field(ge=0)
    stockMinimo: int = Field(ge=0)


class PrediccionResponse(BaseModel):
    demanda_predicha: float
    stock_minimo_recomendado: int
    margen_seguridad: float
    riesgo: str
    reposicion_sugerida: int


def calcular_stock_minimo_recomendado(request: PrediccionRequest, demanda_predicha: float) -> tuple[int, float]:
    """
    Calcula un punto de reposicion dinamico para la siguiente semana.

    - La demanda base proviene del modelo de regresion.
    - El margen de seguridad usa la variabilidad de las ultimas 4 semanas.
    - 1.65 corresponde aproximadamente a un nivel de servicio unilateral del 95 %.

    No es un segundo modelo entrenado: es una regla operativa derivada de la
    prediccion y del historial reciente.
    """
    ultimas_4 = [request.lag_1, request.lag_2, request.lag_3, request.lag_4]
    desviacion = statistics.pstdev(ultimas_4) if len(ultimas_4) > 1 else 0.0
    margen_seguridad = 1.65 * desviacion
    stock_minimo_recomendado = max(0, int(math.ceil(demanda_predicha + margen_seguridad)))
    return stock_minimo_recomendado, margen_seguridad


def clasificar_riesgo(stock_actual: int, stock_minimo_recomendado: int, demanda_predicha: float) -> str:
    if stock_actual < demanda_predicha:
        return "ALTO"
    if stock_actual <= stock_minimo_recomendado:
        return "MEDIO"
    return "BAJO"


@app.get("/health")
def health():
    return {"status": "ok", "modelo": MODEL_PATH.name}


@app.post("/predecir", response_model=PrediccionResponse)
def predecir(request: PrediccionRequest):
    try:
        entrada = pd.DataFrame([{
            "lag_1": request.lag_1,
            "lag_2": request.lag_2,
            "lag_3": request.lag_3,
            "lag_4": request.lag_4,
            "promedio_movil_4": request.promedio_movil_4,
            "promedio_movil_8": request.promedio_movil_8,
            "tendencia_4": request.tendencia_4,
            "descuento_pct": request.descuento_pct,
            "mes": request.mes,
            "semana_del_ano": request.semana_del_ano,
            "categoria": request.categoria,
        }])
        demanda = float(modelo.predict(entrada)[0])
        demanda = max(0.0, demanda)
        stock_minimo_recomendado, margen_seguridad = calcular_stock_minimo_recomendado(request, demanda)
        riesgo = clasificar_riesgo(request.stockActual, stock_minimo_recomendado, demanda)
        reposicion = max(0, stock_minimo_recomendado - request.stockActual)
        return PrediccionResponse(
            demanda_predicha=round(demanda, 2),
            stock_minimo_recomendado=stock_minimo_recomendado,
            margen_seguridad=round(margen_seguridad, 2),
            riesgo=riesgo,
            reposicion_sugerida=reposicion,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"No fue posible generar la prediccion: {exc}") from exc
