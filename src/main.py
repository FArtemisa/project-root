import sys
import os
import yaml
from src.data_loader import load_and_preprocess_data
from src.model_trainer import train_and_save_model

def load_config(config_path="config/params.yaml"):
    """
    Carga y valida el archivo de configuración YAML.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Validación de secciones mínimas requeridas
    for section in ("data", "model", "paths"):
        if section not in config:
            raise KeyError(f"Falta la sección '{section}' en {config_path}")
    
    # Validación de campos críticos dentro de 'data'
    for field in ("raw_path", "test_size", "random_state"):
        if field not in config["data"]:
            raise KeyError(f"Falta el campo '{field}' en data de {config_path}")
    
    # Validación del modelo soportado
    supported = ["RandomForest", "LogisticRegression", "SVM"]
    if config["model"].get("name") not in supported:
        raise ValueError(f"Modelo no soportado. Usa: {supported}")
    
    # Validación de ruta de guardado del modelo
    if "model_save" not in config["paths"]:
        raise KeyError("Falta 'model_save' en la sección paths")
    
    return config

def main():
    try:
        print("Iniciando pipeline de predicción de churn")
        
        # Cargar configuración
        print("\n[1/3] Cargando configuración desde config/params.yaml...")
        config = load_config()
        print("Configuración cargada exitosamente.")
        
        # Preprocesamiento de datos (Data Engineer)
        print("\n[2/3] Ejecutando preprocesamiento de datos...")
        X_train, X_test, y_train, y_test = load_and_preprocess_data(config)
        print(f"Datos listos: Train shape = {X_train.shape}, Test shape = {X_test.shape}")
        
        # Entrenamiento y evaluación del modelo (ML Engineer)
        print("\n[3/3] Entrenando y guardando modelo...")
        metrics = train_and_save_model(X_train, y_train, X_test, y_test, config)
        
        # Mostrar resultados
        print("PIPELINE FINALIZADO")
        print("\nMétricas obtenidas en el conjunto de prueba:")
        print(f"  - Mejor modelo : {metrics['best_model']}")
        print(f"  - Accuracy : {metrics['accuracy']:.4f}")
        print(f"  - Recall   : {metrics['recall']:.4f}")
        print(f"  - F1 Score : {metrics['f1_score']:.4f}")
        print(f"\nModelo guardado en: {config['paths']['model_save']}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()