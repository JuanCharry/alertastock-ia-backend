from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

BASE_DIR = Path(__file__).resolve().parent
DATASET = BASE_DIR / "data" / "AlertaStock_dataset_V2_52_semanas.csv"
MODEL_OUT = BASE_DIR / "modelo_alertastock_regresion.pkl"

VARIABLES_NUMERICAS = [
    "lag_1", "lag_2", "lag_3", "lag_4",
    "promedio_movil_4", "promedio_movil_8",
    "tendencia_4", "descuento_pct", "mes", "semana_del_ano"
]
VARIABLES_CATEGORICAS = ["categoria"]
CARACTERISTICAS = VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS
OBJETIVO = "cantidad_vendida"


def main():
    df = pd.read_csv(DATASET)
    df["semana"] = pd.to_datetime(df["semana"])
    df = df.sort_values(["semana", "productoId"]).reset_index(drop=True)

    semanas = sorted(df["semana"].unique())
    # Modelo final: 44 semanas (train + validacion), dejando 8 para test.
    semanas_train_val = semanas[:44]
    train_val = df[df["semana"].isin(semanas_train_val)].copy()

    X = train_val[CARACTERISTICAS]
    y = train_val[OBJETIVO]

    preprocesador = ColumnTransformer(
        transformers=[
            ("categoricas", OneHotEncoder(handle_unknown="ignore"), VARIABLES_CATEGORICAS),
            ("numericas", "passthrough", VARIABLES_NUMERICAS),
        ]
    )

    modelo = Pipeline(
        steps=[
            ("preprocesamiento", preprocesador),
            ("modelo", LinearRegression()),
        ]
    )

    modelo.fit(X, y)
    joblib.dump(modelo, MODEL_OUT)
    print(f"Modelo guardado en: {MODEL_OUT}")
    print(f"Registros usados: {len(train_val)}")


if __name__ == "__main__":
    main()
