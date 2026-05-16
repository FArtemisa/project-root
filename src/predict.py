"""
Script de predicción para nuevos clientes.

Carga el modelo entrenado y la lista de columnas usadas durante el entrenamiento,
preprocesa un cliente de ejemplo y devuelve la predicción de churn.

Uso:
    python -m src.predict
"""

import os
import sys
import pandas as pd
import joblib
import yaml


def load_config(config_path="config/params.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No se encuentra {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def preprocess_new_customer(customer_data, feature_columns):
    """
    Aplica el mismo preprocesamiento que data_loader.py y alinea las columnas
    con las que se usaron durante el entrenamiento.
    """
    df = pd.DataFrame([customer_data])

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    if df['TotalCharges'].isna().any():
        df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'])
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    df['Partner'] = df['Partner'].map({'Yes': 1, 'No': 0})

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)

    df = df.reindex(columns=feature_columns, fill_value=0)
    return df


def predict_example():
    try:
        config = load_config()
        model_path = config['paths']['model_save']
        features_path = config['paths'].get('features_save', 'models/features.pkl')

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No se encontró el modelo en {model_path}. "
                "Ejecuta 'python -m src.main' primero."
            )
        if not os.path.exists(features_path):
            raise FileNotFoundError(
                f"No se encontró el archivo de features en {features_path}. "
                "Ejecuta 'python -m src.main' primero."
            )

        model = joblib.load(model_path)
        feature_columns = joblib.load(features_path)
        print(f"Modelo cargado desde {model_path}")

        nuevo_cliente = {
            'gender': 'Male',
            'SeniorCitizen': 0,
            'Partner': 'Yes',
            'Dependents': 'No',
            'tenure': 12,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'Fiber optic',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'Yes',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'Yes',
            'StreamingMovies': 'No',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check',
            'MonthlyCharges': 75.5,
            'TotalCharges': '850.5',
        }

        cliente_procesado = preprocess_new_customer(nuevo_cliente, feature_columns)

        prediccion = int(model.predict(cliente_procesado)[0])
        probabilidad = (
            model.predict_proba(cliente_procesado)[0]
            if hasattr(model, "predict_proba")
            else None
        )

        print("\n--- Predicción para nuevo cliente ---")
        print(f"Predicción de Churn: {'Sí' if prediccion == 1 else 'No'}")
        if probabilidad is not None:
            print(f"Probabilidad de abandono: {probabilidad[1]:.4f}")

        return prediccion

    except Exception as e:
        print(f"Error durante la predicción: {e}")
        sys.exit(1)


if __name__ == "__main__":
    predict_example()
