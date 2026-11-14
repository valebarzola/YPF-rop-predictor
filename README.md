# YPF-rop-predictor

Predicción de **ROP (*Rate of Penetration* / velocidad de penetración)**, descubrimiento de regímenes operativos y puesta en producción con monitoreo continuo en pozos de perforación petrolera del bloque Volve (Mar del Norte, Noruega).

## Integrantes

- Daiana Soledad Ruggeri
- Juan Manuel Sosa
- Juan Bensadon
- Valentin Barzola

---

## Síntesis Ejecutiva del Proyecto

- **Origen del dataset:** Utilizamos el dataset público de perforación petrolera de la Universidad de Stavanger correspondiente al yacimiento **NPD Volve (bloque 15/9, Mar del Norte, Noruega)**, compuesto por registros reales de sensores de superficie y fondo en **7 pozos direccionales** (`Pozos 0, 1, 2, 3, 4, 6` destinados al entrenamiento y `Pozo 5` reservado íntegramente a ciegas como pozo *holdout* de validación productiva). Detectamos y corregimos un defecto crítico del repositorio público original: el archivo oficial del Pozo 1 (`F-7`) era en realidad una copia recortada del Pozo 6 (`F-9`), por lo que reconstruimos el Pozo 1 desde cero a partir de sus sensores crudos (`well1_crudo.csv`, 65 columnas y 21.013 filas).
- **Resumen y objetivo propuesto:** El **ROP (*Rate of Penetration*, medido en m/h)** es la velocidad a la que el trépano avanza perforando la roca y constituye un indicador crítico porque el tiempo de equipo en torre representa el mayor costo operativo de un pozo. Nos propusimos como objetivo construir un flujo completo de Ciencia de Datos que permita: *(1)* limpiar y estandarizar las señales crudas sin introducir fugas de información temporal, *(2)* descubrir de forma no supervisada los regímenes físicos de operación y maniobras anómalas, *(3)* predecir el ROP en pozos nuevos evaluando rigurosamente la capacidad de generalización entre pozos distintos, y *(4)* empaquetar el modelo en un `Pipeline` productivo desplegable como API web (`FastAPI` + `Docker`) con monitoreo automático de deriva de datos (*Data Drift* y *Concept Drift*).
- **Descubrimientos del análisis exploratorio y decisiones tomadas:** En el análisis exploratorio descubrimos tres fenómenos determinantes que guiaron nuestras decisiones: primero, que el relleno temporal (`ffill`/`bfill`) repite artificialmente una misma lectura en decenas de metros e infla las métricas, por lo que decidimos trabajar exclusivamente con mediciones reales sin relleno; segundo, que los Pozos 3 y 4 muestreaban entre 4 y 6 veces por metro concentrando el $79,6\%$ de los datos, lo que resolvimos creando un dataset balanceado a **1 fila por metro perforado (`8.473 filas`)**; y tercero, que existía **confusión por profundidad (*depth confounding*)** y multicolinealidad extrema entre `Measured Depth m` y `Hole Depth (TVD) m` ($\text{VIF} > 50$), ya que los únicos pozos profundos de entrenamiento (Pozos 3 y 4) eran más lentos ($\sim 21\text{ m/h}$), lo que engañaba a los modelos al evaluar el Pozo 5 ($32,2\text{ m/h}$).
- **Modelos de aprendizaje no supervisado (`SVD`, `PCA`, `K-Means` y `DBSCAN`):** Aplicando `StandardScaler` sobre las **7 variables operativas puras** (sin ver profundidad, pozo ni ROP), comprobamos la equivalencia exacta entre **SVD truncado** ($X = U \Sigma V^T$) y **PCA**, cuantificando cómo el truncamiento de rango ($B_{m \times r}$) reduce el error de reconstrucción de Frobenius filtrando el ruido de sensores y concentrando el **$67,72\%$ de la varianza en 2 componentes** (`PC1`: geometría/caudal vs. carga mecánica en el trépano; `PC2`: esfuerzo de suspensión y presión de bombeo). Sobre ese espacio, **K-Means ($k = 4$, $\text{Silhouette} = 0,4091$)** descubrió 4 regímenes físicos reales alineados con los tramos constructivos del pozo ($\text{ARI} = 0,4060$), mientras que **DBSCAN (`eps = 0,7`, `min_samples = 10`)** aisló el $0,85\%$ de maniobras atípicas de transición donde la volatilidad del ROP se duplica.
- **Modelos de aprendizaje supervisado y redes neuronales (`MLPRegressor`):** Demostramos cómo el esquema de validación cambia radicalmente el diagnóstico: en un *split* aleatorio 80/20 **XGBoost** alcanza $R^2 = 0,8871$ ($\text{MAE} = 3,00\text{ m/h}$), pero al validar por **bloques contiguos de 100 m** baja a $R^2 = 0,5758$ y en validación cruzada por pozo completo (**LOWO CV**) obliga a regularizar con `GridSearchCV` (`max_depth=4`) y eliminar `Measured Depth m`, logrando pasar en el **Pozo 5 (Holdout)** de $R^2 = -0,1083$ a **$R^2 = +0,1181$ ($\text{MAE} = 11,76\text{ m/h}$)**. En **Redes Neuronales (`MLPRegressor`)**, demostramos el impacto de la función de activación (*vanishing gradient* en sigmoide vs. activación esparsa en `ReLU` y estabilidad de extrapolación en `tanh`), la penalización Ridge ($L_2$, `alpha`) sobre los pesos $W$ sin restringir los sesgos (*bias*), y cómo las arquitecturas profundas memorizan pozos individuales si no se regularizan con `early_stopping`.
- **Puesta en producción (`Pipeline` + `FastAPI` + `Docker`) y monitoreo de *Drift*:** Empaquetamos el preprocesamiento y los modelos en un único objeto sellado (`produccion/modelo_rop_pipeline.joblib`) expuesto mediante una API REST en `produccion/main.py` (`FastAPI` + `Dockerfile` listo para **Hugging Face Spaces**), soportando predicción individual en tiempo real (`/predecir`), predicción por lote (`/predecir_lote`) y auditoría continua (`/monitorear_drift`). Al simular la operación del Pozo 5 en 4 lotes sucesivos de profundidad, detectamos **Data Drift** mediante el test de **Kolmogorov-Smirnov (`K-S`)** y el **Population Stability Index (`PSI` $> 0,20$)** junto con **Concept Drift**, demostrando que la adaptación incremental lote a lote con **`partial_fit` (`v3`)** reduce el error en el Pozo 5 de $\text{MAE} = 14,88\text{ m/h}$ (modelo estático `v1`) a **$\text{MAE} = 11,33\text{ m/h}$ ($\text{RMSE} = 13,72\text{ m/h}$)**.

---

## ¿Qué es el ROP y por qué importa?

**ROP (*Rate of Penetration*)** es la velocidad a la que el trépano avanza perforando la roca, medida en metros por hora (m/h). Es una de las variables más importantes de la operación de perforación porque **el tiempo de taladro es el mayor costo de perforar un pozo**: cada hora de equipo operando cuesta dinero real, avance rápido o lento. Poder predecir el ROP y caracterizar los regímenes operativos a partir de los parámetros de perforación (peso sobre la mecha, rotación, torque, presión de bombeo, etc.) permite:

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
| 5 (F-5) | Prueba final (holdout) | Oficial (`USROP_A 5...csv`), reservado sin explorar | `USROP_A 5 sin relleno.csv` (908 filas, usado únicamente en la evaluación final y monitoreo productivo) |

En resumen: **6 pozos para entrenar (0, 1 propio, 2, 3, 4, 6) y 1 pozo reservado como holdout final (5)**.

---

## Contenido de este repositorio (Pipeline en 6 etapas)

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
   - Comparado contra el **`Baseline (Media)` del entrenamiento** ($\text{MAE} = 15.9627\text{ m/h}$, $R^2 = -1.0295$), **XGBoost reduce el MAE a $11.7301\text{ m/h}$ con MD y $12.1595\text{ m/h}$ sin MD**, mientras que **Random Forest sin MD mejora su $R^2$ de $-0.8461$ a $-0.5233$ y baja su MAE de $13.7163$ a $12.2452\text{ m/h}$**.
   - **Optimización con `GridSearchCV` por pozo (`GroupKFold(n_splits=6)` sobre las 7 variables operativas):** la búsqueda selecciona árboles más someros (`max_depth=4`), reduciendo el MAE en LOWO de **Random Forest a $11.1910\text{ m/h}$ ($R^2 = -0.2832$)**, de **XGBoost a $11.3351\text{ m/h}$ ($R^2 = -0.2909$)** y de **MLP a $14.3889\text{ m/h}$ ($R^2 = -1.0744$)**.
4. **Experimento 3 — Evaluación final en Holdout (Pozo 5): efecto de eliminar `Measured Depth m` y de `GridSearchCV`**:
   - **Con `Measured Depth m` (8 variables)**: los árboles asimilan el Pozo 5 a los únicos pozos profundos de entrenamiento (Pozos 3 y 4, $\sim 21\text{ m/h}$) y subestiman el ROP, dando $R^2 = -0.1083$ y $\text{MAE} = 13.5304\text{ m/h}$.
   - **Sin `Measured Depth m` (7 variables operativas puras)**: **XGBoost pasa a $R^2 = +0.1181$, $\text{RMSE} = 14.3024\text{ m/h}$ y reduce el MAE a $11.7568\text{ m/h}$**, mientras que en el dataset completo **Random Forest alcanza $R^2 = +0.1046$ y $\text{MAE} = 11.7636\text{ m/h}$**.
5. **Experimento 4 — Contraste con Persistencia y prueba de calibración inicial en Pozo 5**:
   - **Baseline de persistencia ($\widehat{\text{ROP}}_t = \text{ROP}_{t-1}$)**: predecir con el ROP del metro anterior en el Pozo 5 alcanza **$R^2 = 0.6708$ ($\text{RMSE} = 8.73\text{ m/h}$ y $\text{MAE} = 4.42\text{ m/h}$)**, motivando las estrategias de aprendizaje incremental por lotes implementadas en la etapa de producción.

---

### Etapa 4 — `ROP_Noruega_No_Supervisado.ipynb`

Notebook de **aprendizaje no supervisado (`SVD`, `PCA`, `K-Means` y `DBSCAN`)** sobre `conjunto_limpio_balanceado.csv` y proyección del `Pozo 5`, entrenando únicamente con las **7 variables operativas de perforación** (reservando `ROP`, `Measured Depth m` y `pozo` como variables externas para interpretar y validar los resultados).

#### Proceso y hallazgos principales

1. **Importancia del escalado (`StandardScaler`)**:
   - Se demuestra que sin escalar, `Average Standpipe Pressure kPa` (en miles de kPa) domina por completo las distancias euclidianas y hace que K-Means corte únicamente por presión de bombeo. Al estandarizar (`StandardScaler`), las 7 variables aportan equilibradamente.
2. **Descomposición en Valores Singulares (SVD), Reconstrucción Truncada ($B_{m \times r}$) y PCA**:
   - Se comprueba numéricamente la equivalencia entre `PCA` (centrado de datos + SVD truncado) y la proyección $Z_r = U_r \Sigma_r$ ($\text{diferencia máxima} = 7.33 \times 10^{-14}$).
   - Se evalúan los **dos criterios de selección del rango $r$**: *(1)* la caída de la **distancia Euclídea / norma de Frobenius ($\|M - M_r\|_F$)** entre la matriz original y la matriz truncada reconstruida, demostrando cómo el truncamiento en $r = 3\text{ o }4$ filtra el ruido de alta frecuencia de los sensores conservando la estructura operativa, y *(2)* el peso relativo acumulado de los valores singulares ($\Sigma$) y varianza explicada ($67,72\%$ en 2 componentes y $82,35\%$ en 3 componentes).
3. **Segmentación de regímenes operativos con K-Means ($k = 4$, $\text{Silhouette} = 0.4091$) y perfil de Silhouette por punto**:
   - Sin ver la profundidad, el pozo ni el ROP, K-Means descubre **4 regímenes reales de perforación** alineados con las secciones constructivas del pozo ($\text{Adjusted Rand Index} = 0.4060$):
     - **Cluster 1 — *Superficial ($17.5\text{''}/12.25\text{''}$) de bajo esfuerzo y alto ROP* ($1.969\text{ filas}, 23.2\%$):** agrupa el 100% de los **Pozos 0, 1 y 6** ($\text{ROP medio} = 41.8\text{ m/h}$).
     - **Cluster 0 — *Intermedio de alto bombeo y bajo WOB* ($3.175\text{ filas}, 37.5\%$):** tramos medios de los Pozos 2, 3 y 4 ($\text{ROP medio} = 24.0\text{ m/h}$).
     - **Cluster 3 — *Profundo ($8.5\text{''}$) estándar de alto torque* ($2.437\text{ filas}, 28.8\%$):** tramos profundos de los Pozos 2, 3 y 4 ($\text{ROP medio} = 21.6\text{ m/h}$).
     - **Cluster 2 — *Profundo ($8.5\text{''}$) de alta energía mecánica (alto WOB y RPM)* ($892\text{ filas}, 10.5\%$):** exclusivo del Pozo 4 en entrenamiento ($\text{ROP medio} = 23.6\text{ m/h}$).
4. **Clustering por densidad con DBSCAN y detección de maniobras atípicas (`-1`)**:
   - Con `eps = 0.7` y `min_samples = 10` se clasifican $8.293$ puntos *Core*, $108$ *Border* y **$72$ puntos *Noise* (`-1`, $0.85\%$)** correspondientes a maniobras de transición donde la dispersión del ROP casi se duplica ($\text{std} = 24.8\text{ m/h}$ vs. $14.4\text{ m/h}$).

---

### Etapa 5 — `ROP_Noruega_Redes_Neuronales.ipynb`

Notebook dedicada al **diseño, diagnóstico interno de activaciones y pesos, regularización, optimización y monitoreo en producción de Redes Neuronales (`MLPRegressor`)** sobre los pozos de Noruega y el **Pozo 5 (Holdout)**.

#### Proceso y hallazgos principales

1. **Efecto del escalado en el descenso por gradiente**:
   - Sin escalar las variables de entrada, la diferencia de órdenes de magnitud estanca el descenso por gradiente ($\text{RMSE Train} = 11.64\text{ m/h}$). Con `StandardScaler` dentro de un `Pipeline`, el error de entrenamiento baja a $6.37\text{ m/h}$ sin fuga de datos.
2. **Funciones de activación (`relu`, `tanh`, `logistic`), *Vanishing Gradient* y *Sparse Activation / Dying ReLU***:
   - Se analiza por qué la activación sigmoidea (`logistic`, derivada máxima $\le 0,25$) sufre atenuación exponencial del gradiente al propagarse hacia atrás (*backpropagation*) frente a `relu` (derivada $1$ para $z > 0$), cuantificando el porcentaje de activaciones esparsas en cero y neuronas inactivas (*Dying ReLU*), así como la ventaja de `tanh` (centrada en cero y acotada en $[-1, 1]$) para evitar extrapolaciones descontroladas en pozos nuevos.
3. **Arquitectura de la red, regularización Ridge ($L_2$: Pesos $W$ vs. Sesgos *Bias* $b$) y `early_stopping`**:
   - Se demuestra cómo al crecer de `(8,)` ($73\text{ parámetros}$) a `(128, 64, 32)` ($11.393\text{ parámetros}$) la brecha de sobreajuste entre entrenamiento y LOWO casi se duplica ($9.26\text{ a }17.66\text{ m/h}$).
   - Se inspeccionan las matrices `coefs_` ($W$) e `intercepts_` ($b$) para demostrar que la regularización Ridge ($L_2$, `alpha`) achica y distribuye uniformemente la norma de los pesos $\|W\|_2$ evitando conexiones dominantes sin llevarlos exactamente a cero, mientras que los sesgos (*bias*) conservan su libertad para ajustar el umbral de activación.
4. **Monitoreo por lotes en producción (`Kolmogorov-Smirnov` y `PSI`) y adaptación con `partial_fit` en el Pozo 5**:
   - Al simular la perforación del Pozo 5 en **4 lotes sucesivos de profundidad**, tanto el test **K-S** ($p < 10^{-50}$) como el **Population Stability Index (`PSI` $> 0.20$)** detectan *Data Drift* acompañado de *Concept Drift*.
   - La actualización incremental lote a lote con **`partial_fit` (`v3`) reduce el error en el Pozo 5 a $\text{MAE} = 11.33\text{ m/h}$ y $\text{RMSE} = 13.72\text{ m/h}$**, superando a la red estática original (`v1`, $\text{MAE} = 14.88\text{ m/h}$) y al reentrenamiento único con el tramo inicial (`v2`, $\text{MAE} = 13.55\text{ m/h}$).

---

### Etapa 6 — Puesta en Producción y Dashboard Interactivo (`produccion/` + `dashboard.html`)

Para llevar el modelo a operación real reduciendo la fricción técnica (*MLOps*), el proyecto incluye:

1. **Servicio Web Productivo con FastAPI y Docker (`produccion/`)**:
   - `produccion/modelo_rop_pipeline.joblib`: paquete sellado e indivisible (`Pipeline` de `scikit-learn` con escalador congelado + modelos supervisados y `K-Means` + distribución de referencia de entrenamiento).
   - `produccion/main.py`: API REST construida con **FastAPI** que incluye interfaz visual interactiva en `/`, documentación automática Swagger UI en `/docs`, predicción individual en tiempo real (`POST /predecir`), predicción por lote (`POST /predecir_lote`) y monitoreo automático de *Data Drift* y *Concept Drift* (`POST /monitorear_drift`).
   - `produccion/requirements.txt` y `produccion/Dockerfile`: configuración lista para desplegar en un contenedor Linux (`python:3.11-slim`, puerto `7860`) en **Hugging Face Spaces** o en cualquier servidor local/nube.
   - `produccion/cliente_prueba.py`: cliente de ejemplo en Python para consumir los endpoints productivos.
2. **Dashboard Autocontenido (`dashboard.html`)**:
   - Aplicación web interactiva en un único archivo HTML sin dependencias externas que integra todas las figuras de las notebooks, explorador de perfiles de pozo por profundidad, simulador de la API en producción en tiempo real, proyecciones `PCA/K-Means` y comparador de modelos.

---

## Requisitos y Ejecución

Instalación de paquetes para las notebooks:
```bash
pip install -r requirements_rop.txt
```

Ejecución local del servicio web en producción (`FastAPI` en puerto `7860`):
```bash
cd produccion
pip install -r requirements.txt
uvicorn main:app --reload --port 7860
```

## Fuente de los datos

Dataset público **Force ML Lithology / Volve — pozos de perforación del bloque 15/9**, Universidad de Stavanger, Mar del Norte, Noruega.
