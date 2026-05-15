import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, recall_score, f1_score

#Selección del modelo
def train_and_save_model(X_train, y_train, X_test, y_test, config):
    model_name = config["model"]["name"]

    print(f"Entrenando el modelo...")
    print(f"Modelo seleccionado: {model_name}")
    print(f"Configuración usada: {config['model']}\n")

    if model_name == "RandomForest":
        model = RandomForestClassifier(
            n_estimators=config["model"].get("n_estimators", 100),
            max_depth=config["model"].get("max_depth", None),
            random_state=config["data"]["random_state"]
        )

    elif model_name == "LogisticRegression":
        model = LogisticRegression(
            max_iter=config["model"].get("max_iter", 1000),
            class_weight=config["model"].get("class_weight", "balanced")
        )

    elif model_name == "SVM":
        model = SVC(
            C=config["model"].get("C", 1.0),
            kernel=config["model"].get("kernel", "rbf")
        )

    else:
        raise ValueError(f"Modelo no soportado: {model_name}")

    #Entrenamiento del modelo
    model.fit(X_train, y_train)

    #Predicciones
    y_pred = model.predict(X_test)

    #Metricas
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics = {
        "accuracy": accuracy,
        "recall": recall,
        "f1_score": f1
    }

    #Guardar modelo
    model_path = config["paths"]["model_save"]

    model_dir = os.path.dirname(model_path)
    if model_dir:
            os.makedirs(model_dir, exist_ok=True)

    joblib.dump(model, model_path)

    print(f"Modelo guardado en: {model_path}")
    print(f"Métricas: {metrics}")

    return metrics
