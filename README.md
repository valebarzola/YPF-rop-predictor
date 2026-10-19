# YPF-rop-predictor

Predicción de **ROP (Rate of Penetration / velocidad de penetración)** en pozos de perforación petrolera, usando el dataset público de la Universidad de Stavanger (bloque 15/9, Mar del Norte - Noruega).

## Integrantes

- Daiana Soledad Ruggeri
- Juan Manuel Sosa
- Juan Bensadon
- Valentin Barzola

## ¿Qué es el ROP y por qué importa?

**ROP (Rate of Penetration)** es la velocidad a la que el trépano avanza perforando la roca, medida en metros por hora (m/h). Es una de las variables más importantes de la operación de perforación porque **el tiempo de taladro es el mayor costo de perforar un pozo**: cada hora de equipo operando cuesta dinero real, avance rápido o lento. Poder predecir el ROP a partir de los parámetros de perforación (peso sobre la mecha, rotación, torque, presión de bombeo, etc.) permite:

- Planificar cuánto va a tardar un pozo nuevo antes de perforarlo.
- Detectar en tiempo real si algo anda mal durante la operación.
- Ajustar los parámetros de perforación para maximizar la velocidad sin dañar el equipo ni la formación.

## El dataset

Dataset público de la Universidad de Stavanger — **NPD Volve / bloque 15/9, Mar del Norte**, con **7 pozos**. La mayoría de los pozos del dataset ya vienen procesados y limpios (`USROP_A 0/2/3/4/6...csv`), pero **el pozo 1 (F-7)** tiene un problema conocido: el archivo oficial `USROP_A 1 N-S_F-7d.csv` contiene en realidad una copia del pozo 6 (F-9), recortada desde otra profundidad — un bug del script público del dataset (`making_USROP.py`), no un pozo distinto. Por esa razón, **el pozo 1 se limpia desde cero, en su forma cruda tal cual sale de los sensores**, en vez de usar el archivo oficial.

### Distribución de pozos del proyecto

| Pozo | Uso | Archivo crudo / base | Salida limpia para modelado |
|---|---|---|---|
| 0, 2, 3, 4 | Entrenamiento | Oficiales (`USROP_A N...csv`) | `USROP_A N sin relleno.csv` → integrados en `conjunto_limpio.csv` y `conjunto_limpio_balanceado.csv` |
| 6 (F-9) | Entrenamiento | Oficial (`USROP_A 6 N-SH_F-9d.csv`) | `USROP_A 6 sin relleno.csv` → integrado en `conjunto_limpio.csv` y `conjunto_limpio_balanceado.csv` |
| **1 (F-7)** | Entrenamiento | **Crudo desde cero (`well1_crudo.csv`)** — nunca el archivo oficial, que es un bug/copia del pozo 6 | `USROP_A 1 limpio.csv` / rehacimiento sin relleno en el conjunto de entrenamiento |
| 5 (F-5) | Prueba final (holdout) | Oficial (`USROP_A 5...csv`), reservado sin explorar | `USROP_A 5 sin relleno.csv` (908 filas, usado únicamente en la evaluación final) |

En resumen: **6 pozos para entrenar (0, 1 propio, 2, 3, 4, 6) y 1 pozo reservado como holdout final (5)**.

---

## Contenido de este repositorio (Pipeline en 3 etapas)

### Etapa 1 — `ROP_Noruega_Pozo1_Crudo.ipynb`

Notebook de **exploración y limpieza de datos desde cero** del Pozo 1 (F-7), partiendo de su archivo crudo (`well1_crudo.csv`, 65 columnas, 21.013 filas, tal cual sale de los sensores del taladro). Es el primer paso del proyecto: entender la estructura física de los sensores y preparar este pozo para que sea compatible con los demás pozos del bloque Volve.

#### Proceso paso a paso

1. **Preparación del entorno y diagnóstico inicial**: carga del CSV crudo, `df.info()`, chequeo de nombres de columna (se decide, con evidencia, **no** normalizarlos para mantener compatibilidad con los pozos oficiales) y verificación de filas duplicadas / columnas sin variabilidad.
2. **Entendimiento de las 65 columnas**: clasificación por grupos temáticos (perforación/superficie, lodo, Gamma Ray, gas/seguridad, resistividad, navegación direccional, vibración, identificadores, etc.) para decidir por criterio de dominio qué grupos tienen relación real con el ROP.
3. **Selección de columnas**: reducción de 65 a **12 columnas relevantes** (profundidad, peso sobre la mecha, torque, RPM, presión de bombeo, hookload, caudal y densidad de lodo, Gamma Ray, diámetro de mecha y el propio ROP).
4. **Exploración de nulos**: diagnóstico de por qué el núcleo de variables de perforación tiene ~96.7% de nulos (varios sistemas de sensores con distinta frecuencia de reporte) y comprobación estadística de que el patrón de faltantes **no es aleatorio (no es MCAR)**.
5. **Correlación con el ROP (datos crudos)**: primera medición de correlación, antes de rellenar, como punto de comparación posterior.
6. **Orden por profundidad y relleno de huecos**:
   - Se detecta y resuelve el **solapamiento de profundidad** entre los dos tramos de mecha del pozo (36 in y 17.5 in), ordenando primero por tramo y luego por profundidad.
   - Se **descarta la sección de 36 in** (incompatible con el resto del dataset y origen de los valores físicamente imposibles).
   - Se **invalidan valores físicamente imposibles** (pesos, caudales, presiones y torques negativos; ROP fuera de rango) **antes** de rellenar, para que el `ffill`/`bfill` no propague datos corruptos.
   - Recién ahí se aplica `ffill`/`bfill` para completar los huecos, respetando el orden temporal/de profundidad real del pozo.
7. **Exploración de la data limpia**: shape final (19.018 filas × 12 columnas, sin nulos).
8. **Correlación con el ROP (datos limpios)**: nueva matriz de correlación y mapa de calor, con interpretación física de cada variable y detección del fenómeno de **confusión por profundidad** (*depth confounding*).
9. **Distribución de variables y detección de outliers** con dos criterios (IQR y Z-score), diferenciando errores de sensor, eventos reales de la operación o artefactos del relleno (`ffill`).
10. **Guardado del dataset limpio** (`USROP_A 1 limpio.csv`), sin sobrescribir nunca los archivos originales.

---

### Etapa 2 — `ROP_Noruega_Resto_Pozos_Entrenamiento_Sin_Relleno.ipynb`

Notebook de **estandarización, limpieza unificada sin relleno artificial y armado del dataset multi-pozo**. A partir de lo aprendido en el Pozo 1, se observó que el relleno `ffill`/`bfill` repite una misma medición real en decenas de filas consecutivas. Por eso, para entrenar modelos sin filas repetidas artificialmente, se aplica una plantilla de reglas comunes sobre las **mediciones reales** de todos los pozos:

#### Proceso paso a paso

1. **Reglas de limpieza comunes para los 7 pozos**:
   - **Regla 1**: Descartar tramos superiores de 26 pulgadas o más (`Diameter >= 660 mm`).
   - **Regla 2**: Ordenar por tramo (`Diameter mm` descendente) y por profundidad medida (`Measured Depth m` ascendente).
   - **Regla 3**: Reemplazar por `NaN` valores físicamente imposibles y lecturas donde el trépano no está perforando (límites operativos en WOB, RPM, torque, presión, caudal, hookload y ROP), conservando únicamente las **filas completas con mediciones reales** en las 10 columnas núcleo.
2. **Limpieza individual pozo por pozo (Pozos 6, 0, 2, 3, 4 y 1)**: análisis de tramos, outliers sobre mediciones reales y correlaciones simples y parciales con el ROP.
3. **Limpieza a ciegas del Pozo 5 (test)**: aplicación exacta de las mismas reglas operativas sin explorar sus correlaciones ni distribuciones para evitar *data leakage*, generando `USROP_A 5 sin relleno.csv` (908 filas).
4. **Comparación entre pozos y problema de frecuencia de muestreo**:
   - Se detecta que los pozos 3 y 4 registran entre 4 y 6 mediciones por metro, concentrando el **79,6%** de las filas totales, mientras que otros pozos registran 1 medición por metro.
   - Se evalúa la **correlación parcial** (descontando la profundidad), observando que el Pozo 3 presenta correlaciones de signo invertido en presión y caudal, y una desviación horizontal pronunciada entre 3.200 y 3.600 m de MD.
   - Se analiza la **multicolinealidad (VIF)**, detectando redundancia extrema entre `Measured Depth m` y `Hole Depth (TVD) m` ($\text{VIF} > 50$).
5. **Generación de los dos datasets de entrenamiento**:
   - `conjunto_limpio.csv` (**34.456 filas**): todas las mediciones reales completas de los 6 pozos de entrenamiento (`0, 1, 2, 3, 4, 6`).
   - `conjunto_limpio_balanceado.csv` (**8.473 filas**): submuestreo a **una fila por metro de profundidad** en cada pozo para equilibrar el peso de cada pozo según los metros perforados y no según la frecuencia del sensor.

---

### Etapa 3 — `ROP_Noruega_Modelado.ipynb`

Notebook de **modelado supervisado, validación intra-pozo por bloques de 100 m, validación cruzada por pozo (LOWO) y evaluación en el pozo holdout (Pozo 5)**.

#### Modelos evaluados
| # | Modelo | Tipo |
|---|---|---|
| 0 | **Baseline (Media / `DummyRegressor`)** | Piso de referencia (predice siempre la media del entrenamiento) |
| 1 | **Linear Regression** | Modelo lineal estándar |
| 2 | **Ridge Regression** | Lineal regularizado ($L_2$) |
| 3 | **Random Forest** | Ensamble de árboles (*bagging*), captura relaciones no lineales |
| 4 | **XGBoost** | *Gradient boosting* de árboles |
| 5 | **MLP (`MLPRegressor`)** | Red neuronal perceptrón multicapa `(128, 64, 32)` con *early stopping* |

#### Flujo experimental y resultados clave

1. **Preprocesamiento y selección de variables**:
   - Exclusión de `pozo` de las variables predictoras (`X`) tanto en validación como en el holdout final (solo se utiliza como identificador de grupo para particionar).
   - Eliminación de `Hole Depth (TVD) m` por redundancia con `Measured Depth m` ($\text{VIF} > 50$).
   - Evaluación con y sin `Measured Depth m` para corregir el fenómeno de *depth confounding* (los únicos pozos profundos de entrenamiento son los pozos lentos 3 y 4).
   - Escalado con `RobustScaler` integrado dentro de un `Pipeline` de Scikit-Learn.
2. **Experimento 1 — Evaluación intra-pozo: Split aleatorio 80/20 vs. Bloques de 100 m de profundidad**:
   - En el split aleatorio 80/20 de filas sueltas, **XGBoost alcanza $R^2 = 0.8871$ ($\text{MAE} = 3.00\text{ m/h}$)** y **Random Forest $R^2 = 0.8586$ ($\text{MAE} = 3.14\text{ m/h}$)** frente a $R^2 \approx 0.40$ de los lineales.
   - Al separar por **bloques contiguos de 100 m de profundidad (`GroupKFold`)**, el $R^2$ baja a **$0.5758$ en XGBoost ($\text{MAE} = 5.85\text{ m/h}$)** y a **$0.5381$ en Random Forest ($\text{MAE} = 6.16\text{ m/h}$)**, demostrando que mezclar filas sueltas consecutivas infla el resultado por autocorrelación espacial.
3. **Experimento 2 — Validación cruzada *Leave-One-Well-Out* (LOWO CV)**:
   - El $R^2$ en LOWO compara contra la media del pozo de validación (desconocida antes de perforar), dando promedios negativos ($R^2 = -0.47$ en XGBoost). Sin embargo, comparado contra el **`Baseline (Media)` del entrenamiento** ($\text{MAE} = 15.96\text{ m/h}$, $R^2 = -1.03$), **XGBoost reduce el MAE a $11.73\text{ m/h}$** (y **Random Forest sin MD baja su MAE de $13.72$ a $12.25\text{ m/h}$**).
   - El desglose por pozo muestra que para XGBoost los folds más adversos son los **pozos 2 y 3 ($R^2 \approx -0.95$)**, mientras que en los **pozos 4 y 6 obtiene $R^2 \approx -0.04\text{ y }+0.02$**, y en los **pozos 0 y 1 ganan los modelos lineales (`Ridge` / `Linear Regression`, $R^2 = +0.13\text{ y }+0.09$)**.
   - Se demuestra además que estandarizar el target por pozo (Z-score) solo mejoraba el $R^2$ aparente cuando se des-normalizaba con la media y desvío del pozo de validación (*fuga de datos*); usando estadísticas del entrenamiento el $R^2$ cae a $-1.70$.
4. **Experimento 3 — Evaluación final en Holdout (Pozo 5): efecto de eliminar `Measured Depth m`**:
   - El Pozo 5 está íntegramente en el tramo de $8.5\text{''}$ ($215.9\text{ mm}$) entre $2.875\text{ y }3.793\text{ m}$ con ROP medio de $32.2\text{ m/h}$.
   - **Con `Measured Depth m` (8 variables)**: los árboles asimilan el Pozo 5 a los únicos pozos profundos de entrenamiento (Pozos 3 y 4, $\sim 21\text{ m/h}$) y subestiman el ROP, dando $R^2 = -0.1083$ y $\text{MAE} = 13.53\text{ m/h}$, casi empatado con el `Baseline (Media)` ($R^2 = -0.0991$, $\text{MAE} = 14.06\text{ m/h}$).
   - **Sin `Measured Depth m` (7 variables operativas puras)**: **XGBoost pasa a $R^2 = +0.1181$, $\text{RMSE} = 14.30\text{ m/h}$ y reduce el MAE a $11.76\text{ m/h}$** en el dataset balanceado (y **Random Forest alcanza $R^2 = +0.1046$ y $\text{MAE} = 11.76\text{ m/h}$** en el dataset completo).
   - Se incluyen gráficos de **Predicho vs. Real**, **Distribución de errores (residuos) de XGBoost**, **Feature Importance** de las 7 variables operativas y el **Perfil de ROP vs. Profundidad medida** en el Pozo 5.
5. **Experimento 4 — Contraste con Persistencia y prueba de calibración inicial en Pozo 5**:
   - **Baseline de persistencia ($\widehat{\text{ROP}}_t = \text{ROP}_{t-1}$)**: predecir con el ROP del metro anterior en el Pozo 5 alcanza **$R^2 = 0.6708$ y $\text{MAE} = 4.42\text{ m/h}$**, superando a los modelos estáticos que no usan historia reciente.
   - **Calibración con los primeros metros (30, 60 y 100 m)**: corregir el sesgo inicial arroja resultados inestables ($\text{MAE}$ en el resto del pozo de $19.14$, $10.79$ y $11.55\text{ m/h}$ respectivamente), evidenciando que un ajuste constante al inicio del tramo no compensa la falta de variables litológicas (`Gamma Ray`).

#### Conclusiones principales del modelado

- **El *depth confounding* perjudica la transferencia entre pozos**: Quitar `Measured Depth m` y `pozo` obliga a los modelos de árboles a apoyarse en la mecánica de perforación (WOB, RPM, torque, caudal, presión) y permite superar al baseline en el Pozo 5 ($R^2 = +0.12$, $\text{MAE} = 11.76\text{ m/h}$).
- **Límite físico por ausencia de litología (`Gamma Ray`) y memoria temporal**: Aunque los 6 pozos tienen rangos operativos comparables tras la limpieza (Notebook 2), el ROP depende fuertemente de la formación atravesada. Para superar al baseline de persistencia en tiempo real o generalizar con mayor precisión a pozos nuevos es indispensable incorporar registros de litología (`Gamma Ray`) o variables autorregresivas del ROP reciente.

---

## Requisitos

Los paquetes utilizados están listados en `requirements_rop.txt`:

```text
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
xgboost
jupyter
```

Instalación rápida:
```bash
pip install -r requirements_rop.txt
```

## Cómo correr el proyecto

1. Clonar el repositorio o abrir los notebooks en [Google Colab](https://colab.research.google.com/) / VS Code.
2. Seguir el orden de los 3 notebooks:
   - `ROP_Noruega_Pozo1_Crudo.ipynb` → limpieza desde cero del Pozo 1 crudo (`well1_crudo.csv`).
   - `ROP_Noruega_Resto_Pozos_Entrenamiento_Sin_Relleno.ipynb` → limpieza sin relleno de los 7 pozos y generación de `conjunto_limpio.csv`, `conjunto_limpio_balanceado.csv` y `USROP_A 5 sin relleno.csv`.
   - `ROP_Noruega_Modelado.ipynb` → entrenamiento, validación LOWO, evaluación en holdout (Pozo 5) y comparación de modelos.

## Fuente de los datos

Dataset público **Force ML Lithology / Volve — pozos de perforación del bloque 15/9**, Universidad de Stavanger, Mar del Norte, Noruega.

