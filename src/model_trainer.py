import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, recall_score, f1_score


def train_and_save_model(X_train, y_train, X_test, y_test, config):

    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name", "RandomForest")
    train_all = model_cfg.get("train_all", True)  # si False solo el model_name
    selection_metric = model_cfg.get("selection_metric", "f1_score") 

    print("Iniciando entrenamiento")
    print(f"Modo: {'Entrenar todos los modelos' if train_all else f'Entrenar solo {model_name}'}")
    print(f"Métrica de selección: {selection_metric}")
    print(f"Configuración usada: {model_cfg}\n")

    # Instanciar modelos según configuración
    models_to_train = {}

    # RandomForest
    rf = RandomForestClassifier(
        n_estimators=model_cfg.get("n_estimators", 100),
        max_depth=model_cfg.get("max_depth", None),
        random_state=config.get("data", {}).get("random_state", 42)
    )
    models_to_train["RandomForest"] = rf

    # LogisticRegression
    lr = LogisticRegression(
        max_iter=model_cfg.get("max_iter", 1000),
        class_weight=model_cfg.get("class_weight", "balanced"),
        solver=model_cfg.get("solver", "lbfgs")
    )
    models_to_train["LogisticRegression"] = lr

    # SVM
    svm = SVC(
        C=model_cfg.get("C", 1.0),
        kernel=model_cfg.get("kernel", "rbf"),
        probability=model_cfg.get("probability", False)
    )
    models_to_train["SVM"] = svm

    # Si no queremos entrenar todos
    if not train_all:
        if model_name not in models_to_train:
            raise ValueError(f"Modelo no soportado: {model_name}")
        models_to_train = {model_name: models_to_train[model_name]}

    # Entrenar y evaluar
    results = {}
    for name, model in models_to_train.items():
        print(f"Entrenando {name}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results[name] = {
            "accuracy": float(accuracy),
            "recall": float(recall),
            "f1_score": float(f1),
            "model_obj": model
        }

        print(f"{name} métricas: accuracy={accuracy:.4f}, recall={recall:.4f}, f1={f1:.4f}")

    # Seleccion del mejor modelo
    if selection_metric not in ("f1_score", "recall", "accuracy"):
        print(f"Aviso: métrica de selección inválida '{selection_metric}', usando 'f1_score' por defecto.")
        selection_metric = "f1_score"

    # Elegir el mejor por la métrica escogida
    best_name = max(results.keys(), key=lambda n: results[n][selection_metric])
    best_entry = results[best_name]

    # Guardar el mejor modelo
    model_path = config.get("paths", {}).get("model_save", "models/model.pkl")
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    joblib.dump(best_entry["model_obj"], model_path)

    # Preparar datos de retorno sin incluir objetos modelo
    all_metrics = {
        name: {
            "accuracy": data["accuracy"],
            "recall": data["recall"],
            "f1_score": data["f1_score"]
        } for name, data in results.items()
    }
    best_metrics = all_metrics[best_name]

    print(f"Mejor modelo: {best_name}")
    print(f"Guardado en: {model_path}")
    print(f"Métricas del mejor: {best_metrics}\n")

    return {
        "best_model": best_name,
        "best_metrics": best_metrics,
        "all_metrics": all_metrics
    }