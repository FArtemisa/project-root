# 📡 Proyecto Colaborativo MLOps: Predicción de Churn (Abandono de Clientes)

## 🎯 Objetivo del Proyecto
Construir un pipeline de Machine Learning modular, reproducible y colaborativo para predecir si un cliente de telecomunicaciones abandonará el servicio (**Churn**).

El proyecto simula un entorno laboral real donde **4 roles especializados** deben integrar su código en un solo repositorio usando Git.

---

## 📂 El Dataset
Todos los equipos trabajarán con el dataset **Telco Customer Churn**.

*   **Fuente:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
*   **Archivo:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
*   **Problema:** Clasificación Binaria (¿El cliente se va? `Yes`/`No`)
*   **Instrucción Importante:**
    1.  Descarguen el CSV.
    2.  Guárdenlo en la carpeta `data/raw/`.
    3.  **NO suban el CSV a Git** (ya está configurado en `.gitignore` para evitar subir archivos pesados). Cada alumno debe descargarlo localmente.

---

## 👥 Roles y Responsabilidades (Equipos de 4)

Cada miembro del equipo es responsable de un módulo específico. Deben definir sus "contratos de interface" (nombres de funciones y tipos de datos que pasan entre módulos) antes de empezar a codificar.

### 1. 👷 Data Engineer (`src/data_loader.py`)
**Tu misión:** Transformar datos brutos y sucios en datos limpios listos para entrenar.

*   **Tareas Críticas:**
    *   Cargar el CSV desde `data/raw/`.
    *   **Limpieza:** La columna `TotalCharges` tiene espacios vacíos `" "` en lugar de nulos. Debes convertirla a numérico y manejar los NaN resultantes (ej. llenar con mediana o 0).
    *   **Preprocesamiento:** Eliminar `customerID`. Codificar variables binarias (`gender`, `Partner`, `Churn`) de Texto a 0/1.
    *   **División:** Separar en Train/Test usando `test_size` y `random_state` definidos en `config/params.yaml`.
*   **Entregable:** Función `load_and_preprocess_data(config)` que retorna `X_train, X_test, y_train, y_test`.

### 2. 🧠 ML Engineer (`src/model_trainer.py`)
**Tu misión:** Experimentar con algoritmos y guardar el mejor modelo.

*   **Tareas Críticas:**
    *   Implementar una "Fábrica de Modelos" que permita elegir entre al menos **dos algoritmos** (ej. `RandomForest` y `SVM` o `LogisticRegression`) según el config.
    *   Entrenar el modelo con los datos recibidos.
    *   Calcular métricas clave: **Accuracy**, **Recall** (crítico para Churn) y **F1-Score**.
    *   Guardar el modelo entrenado en la carpeta `models/` usando `joblib`.
*   **Entregable:** Función `train_and_save_model(X_train, y_train, X_test, y_test, config)` que guarda el `.pkl` y retorna un diccionario de métricas.

### 3. ⚙️ MLOps Engineer (`src/main.py` y `config/`)
**Tu misión:** Orquestar el flujo y gestionar la configuración externa.

*   **Tareas Críticas:**
    *   Crear y mantener `config/params.yaml`. Debe incluir:
        *   Parámetros de datos (`test_size`, `random_state`).
        *   Parámetros del modelo (`model_name`, `n_estimators`, `C`, `kernel`, etc.).
        *   Rutas de salida.
    *   Escribir `src/main.py`: Este script debe importar las funciones del Data Engineer y del ML Engineer y ejecutarlas en orden.
    *   Asegurar que el proyecto corra con el comando: `python -m src.main`.
*   **Entregable:** Un `main.py` funcional que lea el YAML y ejecute el pipeline completo sin errores de importación.

### 4. 🛡️ QA & Production Engineer (`src/predict.py` y `tests/`)
**Tu misión:** Validar que el sistema funcione y preparar la inferencia para nuevos datos.

*   **Tareas Críticas:**
    *   Crear `src/predict.py`: Un script que cargue el modelo guardado (`models/model.pkl`) y permita predecir la clase de un nuevo cliente (ej. pasando una lista de características manualmente).
    *   Manejo de Errores: Si el modelo no existe, el script debe dar un mensaje claro, no un error críptico.
    *   Escribir tests básicos en `tests/test_pipeline.py` (ej. verificar que `load_data` no retorne DataFrames vacíos).
*   **Entregable:** Un script de predicción robusto y al menos 2 tests unitarios pasando.

---

## 🚀 Flujo de Trabajo con Git

1.  **Clonar:** `git clone <url-del-repo-del-equipo>`
2.  **Ramas:** Cada alumno crea su rama:
    *   `git checkout -b feature/data-engineer`
    *   `git checkout -b feature/ml-engineer`
    *   `git checkout -b feature/mlops-engineer`
    *   `git checkout -b feature/qa-engineer`
3.  **Desarrollo:** Trabajen en paralelo. Hagan commits frecuentes.
4.  **Integración:**
    *   Cuando terminen, hagan `git push` de sus ramas.
    *   El **MLOps Engineer** debe crear un Pull Request (o merge) integrando todas las ramas a `main`.
    *   **Resuelvan conflictos juntos** si dos personas tocaron el mismo archivo (ej. `requirements.txt` o `main.py`).
5.  **Prueba Final:** Ejecuten `python -m src.main` en la rama `main`. Si corre, ¡misión cumplida!

---

## 📂 Estructura de Carpetas

```text
churn-mlops-project/
├── config/
│   └── params.yaml          # Configuración centralizada
├── data/
│   ├── raw/                 # WA_Fn-UseC_-Telco-Customer-Churn.csv (NO SUBIR)
│   └── processed/           # (Opcional) Datos limpios
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Rol: Data Engineer
│   ├── model_trainer.py     # Rol: ML Engineer
│   ├── main.py              # Rol: MLOps Engineer
│   └── predict.py           # Rol: QA Engineer
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py     # Rol: QA Engineer
├── models/                  # Modelos .pkl generados (NO SUBIR o subir solo el final)
├── requirements.txt         # Dependencias
├── .gitignore               # Reglas de exclusión
└── README.md                # Este archivo
```

---

## ✅ Checklist de Entrega

*   [x] El comando `python -m src.main` ejecuta todo el pipeline sin errores.
*   [x] El archivo `config/params.yaml` existe y controla los hiperparámetros.
*   [x] Hay al menos 2 modelos diferentes implementados en el código.
*   [x] El script `predict.py` carga el modelo y hace una predicción de ejemplo.
*   [x] El historial de Git muestra contribuciones de los 4 miembros del equipo.
*   [x] El `README.md` final incluye los resultados obtenidos (Accuracy/Recall del mejor modelo).



---

## 📊 Resultados del mejor modelo

Tras entrenar los 3 modelos (selección por `f1_score`) sobre el split estratificado
(`test_size=0.2`, `random_state=42`):

| Modelo               | Accuracy | Recall | F1-Score |
|----------------------|----------|--------|----------|
| RandomForest         | 0.8041   | 0.5187 | 0.5843   |
| LogisticRegression   | 0.7388   | 0.7807 | 0.6134   |
| **SVM** ⭐           | **0.7473** | **0.7674** | **0.6172** |

**Mejor modelo seleccionado: `SVM`** (con `StandardScaler` previo dentro de un `Pipeline`).
- **Accuracy:** 0.7473
- **Recall:**   0.7674  *(crítico para Churn — capturamos ~77% de los clientes que abandonan)*
- **F1-Score:** 0.6172

> El modelo y la lista de columnas usadas durante el entrenamiento se guardan en
> `models/model.pkl` y `models/features.pkl` para que `predict.py` pueda alinear
> correctamente las features de un cliente nuevo.

---

## 🚀 Cómo ejecutar el proyecto

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Colocar el CSV en data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv

# 3. Entrenar el pipeline completo
python -m src.main

# 4. Probar la predicción sobre un cliente de ejemplo
python -m src.predict

# 5. Correr los tests unitarios
pytest tests/ -q
```

---

## 🤖 Contribución de LLM

Cada integrante documenta aquí qué herramienta de IA usó y para qué tarea concreta.

### 👷 Data Engineer — `src/data_loader.py`
- **Herramienta:** ChatGPT
- **Uso:** Usé un LLM para definir la estructura inicial de la función load_and_preprocess_data y para depurar/mejorar la lógica del código (por ejemplo, correcciones en la imputación por mediana y en la forma segura de asignar resultados en pandas).

### 🧠 ML Engineer — `src/model_trainer.py`
- **Herramienta:** Gemini 3 Flash
- **Uso:** Solicité al LLM el desarrollo de un sistema de comparación que evalúa las métricas de todos los modelos entrenados y selecciona automáticamente el ganador basado en una métrica configurable.

### ⚙️ MLOps Engineer — `src/main.py` y `config/params.yaml`
- **Herramienta:** Deepseek
- **Uso:** Validación defensiva del YAML (secciones y campos requeridos) y orquestación
  del pipeline con manejo de errores y trazas claras por etapa. Además de asesoramiento sobre conceptos básicos y consejos para la depuración del proyecto para asegurar que el proyecto pueda ejecutarse correctamente

### 🛡️ QA & Production Engineer — `src/predict.py` y `tests/test_pipeline.py`
- **Herramienta:** ChatGPT
- **Uso:** Use un LLM (chat gpt) para mejorar la estructura de mi código en el test_pipeline.py y pedir sugerencias para hacer más eficiente el predict. Además, realicé consultas relacionadas con el uso de GitHub y la ejecución de comandos en la terminal para la gestión y control del proyecto.
