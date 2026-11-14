---
title: YPF ROP Predictor API
emoji: 🛢️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Servicio Web en Producción y Monitoreo MLOps — Predicción de ROP (Pozos Noruega)

Este directorio contiene los **4 componentes de puesta en producción** necesarios para desplegar el modelo predictivo de **Tasa de Penetración (`ROP`, m/h)** como un servicio web activo (API REST con **FastAPI** y contenedor **Docker**), listo para alojarse en **Hugging Face Spaces** o ejecutarse localmente.

---

## Estructura de Archivos para Producción

1. **`modelo_rop_pipeline.joblib` (El cerebro entrenado y empaquetado):**
   - Contiene el `Pipeline` sellado e indivisible de `scikit-learn` (`StandardScaler` + `RandomForestRegressor` y `MLPRegressor`), junto con los centroides de regímenes operativos (`K-Means`) y la distribución de referencia del conjunto de entrenamiento para auditoría de *Data Drift*.
   - Al estar empaquetado en un `Pipeline`, los datos nuevos en crudo ingresan directamente a `.predict()` aplicando automáticamente las medias y desvíos estándar congelados durante el entrenamiento, evitando fugas de datos (*data leakage*) y código manual repetitivo (*glue code*).
2. **`main.py` (La API con FastAPI):**
   - Carga `modelo_rop_pipeline.joblib` en memoria al iniciar el servicio y expone los siguientes endpoints:
     - **`GET /`**: Interfaz visual interactiva para probar predicciones en tiempo real desde cualquier navegador.
     - **`GET /docs`**: Interfaz automática **Swagger UI** de FastAPI para probar cada endpoint apretando *"Execute"*.
     - **`GET /salud`**: Verifica el estado operativo del pipeline y muestra las medias y desvíos congelados en el `StandardScaler`.
     - **`POST /predecir`** (*Caso 1 — Predicción individual en tiempo real*): Recibe un JSON con las 8 variables de un metro perforado (`Depth`, `WOB`, `SURF_RPM`, `TORQUE`, `FLOW_IN`, `SPP`, `Vp`, `Vs`) y devuelve la `ROP` estimada en `m/h`, el régimen operativo y alertas de valores atípicos ($|z| > 3$).
     - **`POST /predecir_lote`** (*Caso 2 — Predicción por lote / Batch*): Recibe un conjunto de registros de perforación y devuelve las predicciones vectorizadas junto con el resumen estadístico del tramo.
     - **`POST /monitorear_drift`** (*Monitoreo de Datos y Desempeño en Producción*): Compara las variables entrantes contra el set de entrenamiento mediante el test de **Kolmogorov-Smirnov (`K-S`)** y el **Population Stability Index (`PSI`)**, evalúa el desempeño ($R^2$, $MAE$, $RMSE$) para detectar *Concept Drift* y emite un semáforo automático (`ESTABLE`, `ALERTA MODERADA` o `DRIFT CRÍTICO - REENTRENAR MODELO`).
3. **`requirements.txt` (Lista de dependencias):**
   - Especifica las librerías necesarias (`fastapi`, `uvicorn`, `scikit-learn`, `joblib`, `pandas`, `numpy`, `scipy`, `pydantic`).
4. **`Dockerfile` (Receta de empaquetado en contenedor):**
   - Construye una imagen limpia con `python:3.11-slim`, instala las dependencias de `requirements.txt`, copia `main.py` y `modelo_rop_pipeline.joblib`, y deja encendido `FastAPI` escuchando en el puerto **`7860`**.

---

## Paso a Paso para Subirlo a tu Cuenta de Hugging Face Spaces

1. Ingresá a [https://huggingface.co/spaces](https://huggingface.co/spaces) con tu cuenta y hacé clic en **"Create new Space"**.
2. Completá los datos del Space:
   - **Space name:** `ypf-rop-predictor` (o el nombre que prefieras).
   - **Select the Space SDK:** elegí **Docker** (plantilla **Blank**).
   - **Visibility:** **Public** y hacé clic en **"Create Space"**.
3. En la pestaña **"Files"** de tu nuevo Space, hacé clic en **"Add file" -> "Upload files"** y arrastrá los archivos de esta carpeta `produccion/`:
   - `Dockerfile`
   - `requirements.txt`
   - `main.py`
   - `modelo_rop_pipeline.joblib`
   - `README.md`
4. Hacé clic en **"Commit changes to main"**. Hugging Face construirá automáticamente el contenedor Docker y en ~2 minutos el estado pasará a **`Running`** en tu URL pública:
   - **Panel visual interactivo:** `https://tu-usuario-ypf-rop-predictor.hf.space/`
   - **Formulario automático de prueba FastAPI (`/docs`):** `https://tu-usuario-ypf-rop-predictor.hf.space/docs`

---

## Cómo Ejecutarlo Localmente en tu PC

Desde una terminal ubicada dentro de la carpeta `produccion/`:

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 7860
```

Luego abrí en el navegador:
- **Interfaz visual:** [http://127.0.0.1:7860/](http://127.0.0.1:7860/)
- **Documentación interactiva (`/docs`):** [http://127.0.0.1:7860/docs](http://127.0.0.1:7860/docs)
