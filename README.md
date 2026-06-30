
# PIM5 - Proyecto Integrador M5

Proyecto integrador de ciencia de datos para una empresa financiera ficticia
llamada **FitPredict**. Consiste en el desarrollo de un modelo predictivo de
scoring crediticio, simulando un pipeline de MLOps completo: carga de datos,
analisis exploratorio, feature engineering, entrenamiento y evaluacion de
modelos, y monitoreo de data drift.

---

## Caso de negocio

FitPredict es una empresa financiera que otorga prestamos a personas fisicas.
El objetivo del proyecto es predecir si un cliente va a pagar su prestamo a
tiempo (`Pago_atiempo = 1`) o no (`Pago_atiempo = 0`), a partir de variables
demograficas, financieras y crediticias disponibles al momento de la solicitud.

Contar con un modelo de scoring crediticio confiable permite a la empresa:
- Reducir el riesgo de mora en su cartera de prestamos
- Tomar decisiones de otorgamiento de credito de forma mas objetiva y automatizada
- Detectar a tiempo cambios en el perfil de los clientes que puedan afectar
el desempeno del modelo (data drift)

---

## Dataset

El dataset contiene **10.763 registros** de solicitudes de credito, con **22
variables predictoras** entre numericas, categoricas nominales y categoricas
ordinales, y una variable objetivo binaria (`Pago_atiempo`).

Algunas variables clave:
- `tendencia_ingresos`: tendencia de ingresos del cliente (Creciente/Estable/Decreciente/Desconocido)
- `fecha_prestamo`: fecha de otorgamiento del prestamo
- `puntaje_datacredito`: puntaje crediticio externo
- `salario_cliente`: salario declarado
- `huella_consulta`: cantidad de consultas al historial crediticio

---

## Proceso seguido

### 1. Carga y comprension de datos (`Cargar_datos.ipynb`, `comprension_eda.ipynb`)
- Carga del dataset con separador `;` (formato regional)
- Analisis univariado, bivariado y multivariado
- Deteccion y correccion de valores invalidos en `tendencia_ingresos`
  (58 registros con valores numericos fuera de categoria, tratados como Desconocido)
- Eliminacion de bloque de filas corruptas (~indices 7000-7156)
- Correccion de valor atipico de salario (indice 9832)

### 2. Ingenieria de caracteristicas (`ft_engineering.ipynb`)
- Conversion de `fecha_prestamo` a formato numerico AAAAMMDD
- Imputacion de valores faltantes (media para numericas, moda para categoricas)
- Encoding con `OneHotEncoder` para variables categoricas nominales
- Encoding con `OrdinalEncoder` para `tendencia_ingresos`
  (orden: Desconocido=0, Decreciente=1, Estable=2, Creciente=3)
- Dataset transformado: **8.610 filas x 247 columnas** (train), **2.153 x 247** (test)

### 3. Entrenamiento y evaluacion de modelos (`model_training_evaluation.ipynb`)
Se entrenaron y compararon 3 modelos de clasificacion binaria:

| Modelo | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Arbol de Decision | 0.9954 | 0.9995 | 0.9956 | 0.9976 |
| Random Forest | 0.9805 | 0.9809 | 0.9990 | 0.9899 |
| Regresion Logistica | 0.9526 | 0.9526 | 1.0000 | 0.9757 |

**Modelo seleccionado: Arbol de Decision**, por presentar el mejor desempeno
general (mayor F1-score y accuracy).

### 4. Monitoreo de data drift (`model_monitoring.py`)
Aplicacion desarrollada en **Streamlit** que compara la distribucion de los
datos de referencia (train) vs datos nuevos (test) para detectar cambios que
puedan afectar el desempeno del modelo.

**Metricas calculadas:**
- KS Test (Kolmogorov-Smirnov) para variables numericas
- Chi-cuadrado para variables categoricas

**Hallazgos del monitoreo:**
- `huella_consulta`: 🔴 ALERTA — drift significativo detectado (p-value: 0.0351)
- `saldo_principal`: 🟡 PRECAUCION — zona gris (p-value: 0.0827)
- 20 variables restantes: 🟢 OK

La presencia de drift en `huella_consulta` sugiere que el perfil de consultas
crediticias de los clientes nuevos difiere del grupo de entrenamiento, lo que
podria requerir reentrenamiento del modelo con datos mas recientes.

---

## Estructura del repositorio

PIM5/
├── mlops_pipeline/
│   └── src/
│       ├── Cargar_datos.ipynb
│       ├── comprension_eda.ipynb
│       ├── ft_engineering.ipynb
│       ├── model_training_evaluation.ipynb
│       ├── model_monitoring.py
│       └── model_deploy.py
├── Base_de_datos.csv
└── requirements.txt


## Ramas

- `developer`: desarrollo activo
- `certification`: validacion previa a produccion
- `master`: version estable

---

## Stack tecnologico

- **Lenguaje:** Python 3.12
- **Entorno:** Jupyter Notebooks + venv
- **Librerias:** pandas, numpy, scikit-learn, streamlit, plotly, scipy
- **Version control:** Git + GitHub (Gitflow)
- **Infraestructura:** Docker Desktop (WSL2)

