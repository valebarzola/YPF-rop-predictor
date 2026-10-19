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

Notebook de **modelado supervisado, validación cruzada por pozo (LOWO) y evaluación en el pozo holdout (Pozo 5)**.

#### Modelos evaluados
| # | Modelo | Tipo |
|---|---|---|
| 1 | **Linear Regression** | Baseline lineal |
| 2 | **Ridge Regression** | Lineal regularizado ($L_2$) |
| 3 | **Random Forest** | Ensamble de árboles (*bagging*), captura relaciones no lineales |
| 4 | **XGBoost** | *Gradient boosting* de árboles, mejor rendimiento en datos tabulares |
| 5 | **MLP (`MLPRegressor`)** | Red neuronal perceptrón multicapa `(128, 64, 32)` con *early stopping* |

#### Flujo experimental y resultados clave

1. **Preprocesamiento**:
   - Eliminación de `Hole Depth (TVD) m` por alta multicolinealidad con `Measured Depth m` ($\text{VIF} > 50$).
   - Escalado con `RobustScaler` integrado dentro de un `Pipeline` de Scikit-Learn para mitigar el efecto de outliers operativos sin filtrar información entre train y validación.
2. **Experimento 1 — Split aleatorio 80/20 (interpolación intra-pozo)**:
   - Mezclando filas de todos los pozos, **XGBoost alcanza $R^2 = 0.8888$ ($\text{RMSE} = 4.93\text{ m/h}$)** y **Random Forest $R^2 = 0.8586$**, mientras que los modelos lineales se estancan en $R^2 \approx 0.40$.
   - Confirma que la relación física entre parámetros operativos (WOB, RPM, torque, presión, caudal) y ROP es fuertemente **no lineal** y estadísticamente aprendible dentro de formaciones conocidas.
3. **Experimento 2 — Validación cruzada *Leave-One-Well-Out* (LOWO CV)**:
   - Entrenando con 5 pozos y evaluando en el pozo restante no visto, todos los modelos pasan a obtener $R^2$ promedio negativo (**XGBoost $R^2 = -0.41$** en balanceado y **$-0.52$** en completo).
   - El desglose por fold muestra que el error proviene de la **heterogeneidad geológica entre pozos**: el ROP medio varía un 151% entre pozos lentos (Pozo 4: $20.7\text{ m/h}$, Pozo 3: $22.2\text{ m/h}$) y pozos rápidos (Pozo 6: $51.9\text{ m/h}$).
4. **Experimento 3 — LOWO CV con normalización Z-score por pozo**:
   - Al estandarizar el ROP por pozo ($\mu = 0, \sigma = 1$) para remover el efecto de escala del ROP base de cada formación, el $R^2$ medio de XGBoost mejora de $-0.41$ a **$-0.19$** ($\text{RMSE} = 13.89\text{ m/h}$), demostrando que parte del error cross-well es un desplazamiento de escala, pero que también cambian las relaciones operativas entre formaciones.
5. **Experimento 4 — Evaluación final en Holdout (Pozo 5) y comparación Balanceado vs. Completo**:
   - En el Pozo 5 (nunca visto en el entrenamiento), **XGBoost obtiene el mejor desempeño tanto en el dataset balanceado ($R^2 = -0.1039$, $\text{RMSE} = 16.00\text{ m/h}$, $\text{MAE} = 13.82\text{ m/h}$) como en el completo ($R^2 = -0.2061$, $\text{RMSE} = 16.73\text{ m/h}$)**.
   - El **dataset balanceado supera al completo en XGBoost**, confirmando que equilibrar la frecuencia de muestreo a 1 fila por metro evita que los pozos 3 y 4 sobredimensionen el entrenamiento.
   - Se incluyen gráficos de **Predicho vs. Real**, **Feature Importance** (Random Forest y XGBoost) y el **Perfil de ROP vs. Profundidad medida** a lo largo de todo el Pozo 5.

#### Conclusiones principales del modelado

- **Memorizar no es generalizar**: El contraste entre el split 80/20 ($R^2 = 0.89$) y el LOWO ($R^2 = -0.19$) demuestra que particionar filas consecutivas al azar sobreestima el rendimiento por interpolación; la validación por pozo completo (LOWO) es la métrica honesta para evaluar transferencia a un pozo nuevo.
- **Implicación práctica en operación**: Los modelos basados en árboles (especialmente **XGBoost**) son aptos para **optimización en tiempo real dentro de pozos o formaciones conocidas**, mientras que para un pozo nuevo requieren una etapa de **calibración inicial con los primeros metros perforados** para ajustar el nivel base de ROP de la formación.

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

