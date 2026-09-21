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

| Pozo | Uso | Archivo |
|---|---|---|
| 0, 2, 3, 4 | Entrenamiento | Oficiales, ya procesados por el dataset |
| 6 (F-9) | Entrenamiento | Oficial (`USROP_A 6 N-SH_F-9d.csv`), válido tal cual |
| **1 (F-7)** | Entrenamiento | **Limpieza propia desde cero** (este notebook) — nunca el archivo oficial, que es un bug/copia del pozo 6 |
| 5 (F-5) | Prueba final (test) | Oficial, reservado sin tocar hasta evaluar el modelo entrenado |

En resumen: **6 pozos para entrenar (0, 1 propio, 2, 3, 4, 6) y 1 pozo reservado para test (5)**.

## Contenido de este repositorio

### `ROP_Noruega_Pozo1_Crudo.ipynb`

Notebook de **limpieza de datos desde cero** del Pozo 1 (F-7), partiendo de su archivo crudo (65 columnas, 21.013 filas, tal cual sale de los sensores del taladro). Es el primer paso del pipeline del proyecto: preparar este pozo para que sea compatible y combinable con los demás pozos ya procesados del dataset.

#### Proceso paso a paso

1. **Preparación del entorno y diagnóstico inicial**: carga del CSV crudo, `df.info()`, chequeo de nombres de columna (se decide, con evidencia, **no** normalizarlos para mantener compatibilidad con los pozos oficiales) y verificación de filas duplicadas / columnas sin variabilidad.

2. **Entendimiento de las 65 columnas**: clasificación por grupos temáticos (perforación/superficie, lodo, Gamma Ray, gas/seguridad, resistividad, navegación direccional, vibración, identificadores, etc.) para decidir por criterio de dominio qué grupos tienen relación real con el ROP, antes de calcular ninguna estadística.

3. **Selección de columnas**: reducción de 65 a **12 columnas relevantes** (profundidad, peso sobre la mecha, torque, RPM, presión de bombeo, hookload, caudal y densidad de lodo, Gamma Ray, diámetro de mecha y el propio ROP).

4. **Exploración de nulos**: diagnóstico de por qué el núcleo de variables de perforación tiene ~96.7% de nulos (varios sistemas de sensores con distinta frecuencia de reporte) y comprobación estadística de que el patrón de faltantes **no es aleatorio (no es MCAR)**.

5. **Correlación con el ROP (datos crudos)**: primera medición de correlación, antes de rellenar, como punto de comparación posterior.

6. **Orden por profundidad y relleno de huecos**:
   - Se detecta y resuelve el **solapamiento de profundidad** entre los dos tramos de mecha del pozo (36 in y 17.5 in), ordenando primero por tramo y luego por profundidad.
   - Se **descarta la sección de 36 in** (incompatible con el resto del dataset y origen de los valores físicamente imposibles).
   - Se **invalidan valores físicamente imposibles** (pesos, caudales, presiones y torques negativos; ROP fuera de rango) **antes** de rellenar, para que el `ffill`/`bfill` no propague datos corruptos.
   - Recién ahí se aplica `ffill`/`bfill` para completar los huecos, respetando el orden temporal/de profundidad real del pozo.

7. **Exploración de la data limpia**: shape final (19.018 filas × 12 columnas, sin nulos).

8. **Correlación con el ROP (datos limpios)**: nueva matriz de correlación y mapa de calor, con interpretación física de cada variable y detección del fenómeno de **confusión por profundidad** (*depth confounding*) — casi todas las variables que correlacionan fuerte con el ROP también correlacionan fuerte con la profundidad, excepto `Weight on Bit`.

9. **Distribución de variables** (histogramas) para elegir el método de detección de outliers más adecuado.

10. **Detección e interpretación de outliers** con dos criterios (IQR y Z-score), diferenciando outliers que son error de sensor, evento real de la operación, o artefacto del método de relleno (`ffill`).

11. **Estadística descriptiva final** y valores extremos de ROP ya validados.

12. **Guardado del dataset limpio** (`USROP_A 1 limpio.csv`), sin sobrescribir nunca los archivos originales.

13. **Próximos pasos** para continuar el proyecto (ver abajo).

## Decisiones de limpieza destacadas

- **No se normalizan los nombres de columna**: se comprobó que no tienen problemas de formato y normalizarlos rompería la compatibilidad con los pozos ya procesados del dataset.
- **`ffill`/`bfill` en vez de eliminar filas o imputar con media/mediana**: eliminar filas con algún nulo dejaría solo 7 de 21.013 filas; imputar con una medida global aplanaría la tendencia real del pozo con la profundidad. `ffill`/`bfill` respeta que los datos están ordenados por profundidad y es el mismo criterio usado por el script oficial del dataset.
- **Orden correcto de las operaciones**: ordenar por tramo y profundidad → descartar la sección de 36 in → invalidar valores imposibles → recién ahí rellenar. Se demuestra con evidencia que invertir este orden (por ejemplo, rellenar antes de filtrar) contamina el dataset (un solo ROP corrupto de 7059 m/h se propagó a 159 filas vía `ffill` e invirtió el signo de toda la matriz de correlación).
- **`RobustScaler`** (no `StandardScaler` ni `MinMaxScaler`) queda seleccionado para la etapa de escalado, por la presencia de distribuciones sesgadas y outliers reales en varias variables.

## Próximos pasos del proyecto

- Combinar el pozo 1 (limpio, este notebook) con los pozos oficiales 0, 2, 3, 4 y 6 en un único dataset de entrenamiento.
- Mantener el pozo 5 reservado, sin tocar, como test final del modelo.
- Escalar las variables **recién sobre el dataset combinado** (nunca sobre un pozo aislado), usando `RobustScaler`.
- Entrenar y evaluar el modelo de predicción de ROP, considerando que el pozo 1 aporta solo ~638 valores realmente medidos (el resto son copias por `ffill`) — recomendable evaluar por bloques de profundidad y no por fila suelta al azar.

## Requisitos

```
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
```

## Cómo correr el notebook

1. Clonar el repositorio o abrir el notebook directamente en [Google Colab](https://colab.research.google.com/).
2. Instalar las dependencias si se corre localmente: `pip install pandas numpy matplotlib seaborn scipy scikit-learn`.
3. Ejecutar las celdas en orden — cada sección depende de las decisiones tomadas en la anterior (ver "Proceso paso a paso").
4. El notebook genera como salida `USROP_A 1 limpio.csv`, listo para combinarse con los demás pozos del dataset.

## Fuente de los datos

Dataset público **Force ML Lithology / Volve — pozos de perforación del bloque 15/9**, Universidad de Stavanger, Mar del Norte, Noruega.
