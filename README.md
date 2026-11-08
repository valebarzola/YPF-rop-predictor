# YPF-rop-predictor

Predicción de **ROP (Rate of Penetration / velocidad de penetración)** y descubrimiento de regímenes operativos en pozos de perforación petrolera, usando el dataset público de la Universidad de Stavanger (bloque 15/9, Mar del Norte - Noruega).

## Integrantes

- Daiana Soledad Ruggeri
- Juan Manuel Sosa
- Juan Bensadon
- Valentin Barzola

## ¿Qué es el ROP y por qué importa?

**ROP (Rate of Penetration)** es la velocidad a la que el trépano avanza perforando la roca, medida en metros por hora (m/h). Es una de las variables más importantes de la operación de perforación porque **el tiempo de taladro es el mayor costo de perforar un pozo**: cada hora de equipo operando cuesta dinero real, avance rápido o lento. Poder predecir el ROP y caracterizar los regímenes operativos a partir de los parámetros de perforación (peso sobre la mecha, rotación, torque, presión de bombeo, etc.) permite:

- Planificar cuánto va a tardar un pozo nuevo antes de perforarlo.
- Detectar en tiempo real si algo anda mal o si ocurre una maniobra atípica durante la operación.
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

## Contenido de este repositorio (Pipeline en 5 etapas)

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

Notebook de **modelado supervisado, validación intra-pozo por bloques de 100 m, validación cruzada por pozo (LOWO), optimización de hiperparámetros con `GridSearchCV` y evaluación en el pozo holdout (Pozo 5)**.

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
   - En el split aleatorio 80/20 de filas sueltas, **XGBoost alcanza $R^2_{\text{train}} = 0.9530$ y $R^2_{\text{test}} = 0.8871$ ($\text{MAE} = 3.00\text{ m/h}$)** y **Random Forest $R^2_{\text{train}} = 0.9148$ y $R^2_{\text{test}} = 0.8586$ ($\text{MAE} = 3.14\text{ m/h}$)** frente a $R^2 \approx 0.3968$ de los lineales y $\text{MAE} = 11.52\text{ m/h}$ del `Baseline (Media)`.
   - Al separar por **bloques contiguos de 100 m de profundidad (`GroupKFold`)**, el $R^2$ baja a **$0.5758$ en XGBoost ($\text{MAE} = 5.85\text{ m/h}$)** y a **$0.5381$ en Random Forest ($\text{MAE} = 6.16\text{ m/h}$)**, demostrando que mezclar filas sueltas consecutivas infla el resultado por autocorrelación espacial.
3. **Experimento 2 — Validación cruzada *Leave-One-Well-Out* (LOWO CV) y optimización de hiperparámetros con `GridSearchCV`**:
   - El $R^2$ en LOWO compara contra la media del pozo de validación (desconocida antes de perforar), dando promedios negativos ($R^2 = -0.4710$ con MD y $-0.4708$ sin MD en XGBoost base). Sin embargo, comparado contra el **`Baseline (Media)` del entrenamiento** ($\text{MAE} = 15.9627\text{ m/h}$, $R^2 = -1.0295$), **XGBoost reduce el MAE a $11.7301\text{ m/h}$ con MD y $12.1595\text{ m/h}$ sin MD**, mientras que **Random Forest sin MD mejora su $R^2$ de $-0.8461$ a $-0.5233$ y baja su MAE de $13.7163$ a $12.2452\text{ m/h}$**.
   - El desglose por pozo (con MD) muestra que en **XGBoost** los peores $R^2$ ocurren en el **Pozo 0 ($-1.3089$, $\text{MAE} = 13.76\text{ m/h}$)**, **Pozo 2 ($-0.9586$, mayor $\text{MAE} = 14.78\text{ m/h}$)** y **Pozo 3 ($-0.7795$, $\text{MAE} = 9.72\text{ m/h}$)**, mientras que obtiene $R^2$ positivo en el **Pozo 1 ($+0.0313$)**, **Pozo 4 ($+0.0662$, menor $\text{MAE} = 7.45\text{ m/h}$)** y **Pozo 6 ($+0.1233$)**. En los **Pozos 0 y 1 ganan los modelos lineales** (`Linear Regression` y `Ridge`, $R^2 = +0.1299\text{ / }+0.1298$ en Pozo 0 y $+0.0947\text{ / }+0.0945$ en Pozo 1). Al quitar MD, la dispersión entre pozos de XGBoost cae casi a la mitad ($R^2\text{ std}$ de $0.6211$ a $0.3501$, mejorando en el Pozo 0 a $-0.6055$).
   - **Optimización con `GridSearchCV` por pozo (`GroupKFold(n_splits=6)` sobre las 7 variables operativas):** la búsqueda selecciona árboles más someros (`max_depth=4`), reduciendo el MAE en LOWO de **Random Forest a $11.1910\text{ m/h}$ ($R^2 = -0.2832$)**, de **XGBoost a $11.3351\text{ m/h}$ ($R^2 = -0.2909$)** y de **MLP a $14.3889\text{ m/h}$ ($R^2 = -1.0744$)**.
4. **Experimento 3 — Evaluación final en Holdout (Pozo 5): efecto de eliminar `Measured Depth m` y de `GridSearchCV`**:
   - El Pozo 5 está íntegramente en el tramo de $8.5\text{''}$ ($215.9\text{ mm}$) entre $2.875\text{ y }3.793\text{ m}$ con ROP medio de $32.2\text{ m/h}$.
   - **Con `Measured Depth m` (8 variables)**: los árboles asimilan el Pozo 5 a los únicos pozos profundos de entrenamiento (Pozos 3 y 4, $\sim 21\text{ m/h}$) y subestiman el ROP, dando $R^2 = -0.1083$ y $\text{MAE} = 13.5304\text{ m/h}$, casi empatado con el `Baseline (Media)` ($R^2 = -0.0991$, $\text{MAE} = 14.0641\text{ m/h}$).
   - **Sin `Measured Depth m` (7 variables operativas puras)**: **XGBoost pasa a $R^2 = +0.1181$, $\text{RMSE} = 14.3024\text{ m/h}$ y reduce el MAE a $11.7568\text{ m/h}$** con parámetros fijos (y alcanza **$R^2 = +0.0954$ y $\text{MAE} = 12.1654\text{ m/h}$** con `GridSearchCV`), mientras que en el dataset completo **Random Forest alcanza $R^2 = +0.1046$ y $\text{MAE} = 11.7636\text{ m/h}$**.
   - Se incluyen gráficos de **Predicho vs. Real**, **Distribución de errores (residuos) de XGBoost**, **Feature Importance** de las 7 variables operativas y el **Perfil de ROP vs. Profundidad medida** en el Pozo 5.
5. **Experimento 4 — Contraste con Persistencia y prueba de calibración inicial en Pozo 5**:
   - **Baseline de persistencia ($\widehat{\text{ROP}}_t = \text{ROP}_{t-1}$)**: predecir con el ROP del metro anterior en el Pozo 5 alcanza **$R^2 = 0.6708$ ($\text{RMSE} = 8.73\text{ m/h}$ y $\text{MAE} = 4.42\text{ m/h}$)**, superando a los modelos estáticos que no usan historia reciente.
   - **Calibración con los primeros metros (30, 60 y 100 m)**: corregir el sesgo inicial arroja resultados inestables ($\text{MAE}$ en el resto del pozo de $19.14$, $10.79$ y $11.55\text{ m/h}$ respectivamente), evidenciando que un ajuste constante al inicio del tramo no compensa la falta de variables litológicas (`Gamma Ray`).

---

### Etapa 4 — `ROP_Noruega_No_Supervisado.ipynb`

Notebook de **aprendizaje no supervisado (`SVD`, `PCA`, `K-Means` y `DBSCAN`)** sobre `conjunto_limpio_balanceado.csv` y proyección del `Pozo 5`, entrenando únicamente con las **7 variables operativas de perforación** (reservando `ROP`, `Measured Depth m` y `pozo` como variables externas para interpretar y validar los resultados).

#### Proceso y hallazgos principales

1. **Importancia del escalado (`StandardScaler`)**:
   - Se demuestra que sin escalar, `Average Standpipe Pressure kPa` (en miles de kPa) domina por completo las distancias euclidianas y hace que K-Means corte únicamente por presión de bombeo. Al estandarizar (`StandardScaler`), las 7 variables aportan equilibradamente.
2. **Descomposición en Valores Singulares (SVD) y Análisis de Componentes Principales (PCA)**:
   - Se comprueba numéricamente la equivalencia entre la proyección SVD truncada ($Z_r = U_r \Sigma_r$) y PCA ($\text{diferencia máxima} = 7.33 \times 10^{-14}$), cuantificando el error de reconstrucción de Frobenius ($\|X - B_r\|_F$) desde $r=1$ hasta $r=7$.
   - Los dos primeros componentes retienen el **$67.72\%$ de la varianza total** (`PC1 = 41.85%`, `PC2 = 25.87%`, y con `PC3` el $82.35\%$).
   - **`PC1`** opone geometría/hidráulica (`Diameter mm` $+0.532$ y `Mud Flow In` $+0.529$) contra carga mecánica en el trépano (`Torque` $-0.422$, `WOB` $-0.415$, `RPM` $-0.238$).
   - **`PC2`** representa el esfuerzo de suspensión y bombeo (`Hookload` $+0.662$ y `Standpipe Pressure` $+0.614$).
3. **Segmentación de regímenes operativos con K-Means ($k = 4$, $\text{Silhouette} = 0.4091$) y perfil de Silhouette por punto**:
   - El análisis de `silhouette_samples` verifica la cohesión interna de cada grupo (con apenas $0.00\%$ a $0.37\%$ de observaciones con silueta negativa).
   - Sin ver la profundidad, el pozo ni el ROP, K-Means descubre **4 regímenes reales de perforación** fuertemente alineados con las secciones constructivas del pozo ($\text{Adjusted Rand Index} = 0.4060$ frente al tramo de diámetro):
     - **Cluster 1 — *Superficial ($17.5\text{''}/12.25\text{''}$) de bajo esfuerzo y alto ROP* ($1.969\text{ filas}, 23.2\%$):** agrupa el 100% de los **Pozos 0, 1 y 6** ($\text{MD media} = 793.7\text{ m}$) y duplica el avance del resto del yacimiento (**$\text{ROP medio} = 41.8\text{ m/h}$**).
     - **Cluster 0 — *Intermedio de alto bombeo y bajo WOB* ($3.175\text{ filas}, 37.5\%$):** tramos medios de los Pozos 2, 3 y 4 ($\text{MD media} = 1.971.1\text{ m}$, **$\text{ROP medio} = 24.0\text{ m/h}$**).
     - **Cluster 3 — *Profundo ($8.5\text{''}$) estándar de alto torque* ($2.437\text{ filas}, 28.8\%$):** tramos profundos de los Pozos 2, 3 y 4 ($\text{MD media} = 3.363.2\text{ m}$, **$\text{ROP medio} = 21.6\text{ m/h}$**).
     - **Cluster 2 — *Profundo ($8.5\text{''}$) de alta energía mecánica (alto WOB y RPM)* ($892\text{ filas}, 10.5\%$):** exclusivo del Pozo 4 en entrenamiento ($\text{MD media} = 3.081.6\text{ m}$, **$\text{ROP medio} = 23.6\text{ m/h}$**).
   - **Proyección del Pozo 5 (Holdout):** el 100% de las 908 filas del Pozo 5 cae en los regímenes profundos **C3 ($553\text{ filas}$) y C2 ($355\text{ filas}$)**, pero con un ROP real más alto ($35.3\text{ y }27.3\text{ m/h}$) por mayor perforabilidad de la roca.
4. **Clustering por densidad con DBSCAN y detección de maniobras atípicas (`-1`)**:
   - Usando el gráfico de distancias al 10.° vecino (`NearestNeighbors`), con `eps = 0.7` y `min_samples = 10` se clasifican $8.293$ puntos *Core*, $108$ *Border* y **$72$ puntos *Noise* (`-1`, $0.85\%$)**.
   - Los puntos de ruido corresponden a transiciones operativas con baja rotación ($\text{RPM media} = 85.9$ vs. $156.0$) y baja presión donde la dispersión del ROP casi se duplica ($\text{std} = 24.8\text{ m/h}$ vs. $14.4\text{ m/h}$).

---

### Etapa 5 — `ROP_Noruega_Redes_Neuronales.ipynb`

Notebook dedicada al **diseño, regularización, optimización y monitoreo en producción de Redes Neuronales (`MLPRegressor`)** sobre los pozos de Noruega y el **Pozo 5 (Holdout)**.

#### Proceso y hallazgos principales

1. **Efecto del escalado en el descenso por gradiente**:
   - Sin escalar las variables de entrada, la diferencia de órdenes de magnitud entre `Standpipe Pressure kPa` y `WOB kkgf` estanca el entrenamiento ($\text{RMSE Train} = 11.64\text{ m/h}$). Con `StandardScaler` dentro de un `Pipeline`, el error de entrenamiento baja a $6.37\text{ m/h}$ sin fuga de datos entre pozos.
2. **Funciones de activación (`relu`, `tanh`, `logistic`)**:
   - Si bien `relu` y `tanh` convergen más rápido en entrenamiento que `logistic`, al extrapolar a pozos nuevos (**LOWO** y **Pozo 5**) las funciones de activación acotadas (`tanh` y `logistic`) evitan las extrapolaciones lineales descontroladas de `relu` ante desplazamientos en los sensores.
3. **Arquitectura de la red y brecha de sobreajuste (*overfitting*)**:
   - Al incrementar el tamaño de la red desde `(8,)` ($73\text{ parámetros}$) hasta `(128, 64, 32)` ($11.393\text{ parámetros}$), el `RMSE Train` baja de $9.26$ a $5.61\text{ m/h}$, pero el `RMSE LOWO` empeora de $18.52$ a $23.27\text{ m/h}$ (la brecha de sobreajuste casi se duplica de $9.26$ a $17.66\text{ m/h}$).
4. **Regularización $L_2$ (`alpha`), `early_stopping` y `GridSearchCV` por pozo**:
   - Combinando penalización $L_2$ alta (`alpha = 5.0`), parada temprana (`early_stopping=True`), arquitectura moderada `(64, 32)` y activación `tanh`, `GridSearchCV` reduce el error en **LOWO a $\text{RMSE} = 17.63\text{ m/h}$ y $\text{MAE} = 14.18\text{ m/h}$**, y en el **Pozo 5 (Holdout) a $\text{RMSE} = 16.46\text{ m/h}$ y $\text{MAE} = 14.38\text{ m/h}$** (frente a $\text{MAE} = 18.23\text{ m/h}$ de la red profunda sin regularizar).
5. **Monitoreo por lotes en producción (`Kolmogorov-Smirnov` y `PSI`) y adaptación con `partial_fit` en el Pozo 5**:
   - Al simular la perforación del Pozo 5 en **4 lotes sucesivos de profundidad**, tanto el test **K-S** ($p < 10^{-50}$) como el **Population Stability Index (`PSI` $> 0.20$)** detectan *Data Drift* en las variables operativas, acompañado de *Concept Drift* en los tramos profundos.
   - Evaluando sobre los lotes prospectivos del Pozo 5 (`Lotes 1 a 3`), la actualización incremental lote a lote con **`partial_fit` (`v3`) reduce el error a $\text{MAE} = 11.33\text{ m/h}$ y $\text{RMSE} = 13.72\text{ m/h}$**, superando tanto a la red estática original (`v1`, $\text{MAE} = 14.88\text{ m/h}$) como al reentrenamiento con el tramo inicial (`v2`, $\text{MAE} = 13.55\text{ m/h}$).

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
2. Seguir el orden de los 5 notebooks:
   - `ROP_Noruega_Pozo1_Crudo.ipynb` → limpieza desde cero del Pozo 1 crudo (`well1_crudo.csv`).
   - `ROP_Noruega_Resto_Pozos_Entrenamiento_Sin_Relleno.ipynb` → limpieza sin relleno de los 7 pozos y generación de `conjunto_limpio.csv`, `conjunto_limpio_balanceado.csv` y `USROP_A 5 sin relleno.csv`.
   - `ROP_Noruega_Modelado.ipynb` → entrenamiento supervisado, validación LOWO, optimización con `GridSearchCV`, evaluación en holdout (Pozo 5) y comparación de modelos.
   - `ROP_Noruega_No_Supervisado.ipynb` → descomposición `SVD`, reducción de dimensionalidad (`PCA`), segmentación de regímenes operativos (`K-Means` + `Silhouette`) y detección de anomalías (`DBSCAN`).
   - `ROP_Noruega_Redes_Neuronales.ipynb` → análisis profundo de redes neuronales (`MLPRegressor`), regularización, puesta en producción, monitoreo de *Data/Concept Drift* (`K-S` y `PSI`) y aprendizaje incremental (`partial_fit`) en el Pozo 5.

## Fuente de los datos

Dataset público **Force ML Lithology / Volve — pozos de perforación del bloque 15/9**, Universidad de Stavanger, Mar del Norte, Noruega.
