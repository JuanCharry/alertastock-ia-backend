# Backend IA - AlertaStock

## Ejecutar

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

- Salud: `GET /health`
- Predicción: `POST /predecir`

El Android Emulator usa `http://10.0.2.2:8000/` para conectarse al computador anfitrión.
En un teléfono físico se debe cambiar `IA_BASE_URL` en `app/build.gradle.kts` por la IP LAN del computador y ambos deben estar en la misma red.

## Reentrenar

```bash
python train_model.py
```

El script usa `data/AlertaStock_dataset_V2_52_semanas.csv` y el mismo pipeline de OneHotEncoder + Regresión Lineal del experimento final.
