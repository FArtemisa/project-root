import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, f1_score


def _build_models(model_cfg, random_state):
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=model_cfg.get("n_estimators", 100),
            max_depth=model_cfg.get("max_depth", None),
            random_state=random_state,
        ),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=model_cfg.get("max_iter", 1000),
                class_weight=model_cfg.get("class_weight", "balanced"),
                solver=model_cfg.get("solver", "lbfgs"),
            )),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                C=model_cfg.get("C", 1.0),
                kernel=model_cfg.get("kernel", "rbf"),
                probability=model_cfg.get("probability", True),
                class_weight=model_cfg.get("class_weight", "balanced"),
            )),
        ]),
    }


def train_and_save_model(X_train, y_train, X_test, y_test, config):
    """
    Entrena uno o más modelos, guarda el mejor en disco y retorna sus métricas.

    Args:
        X_train, y_train, X_test, y_test: datos del Data Engineer.
        config (dict): configuración cargada desde params.yaml.

    Returns:
        dict con keys: accuracy, recall, f1_score (del mejor modelo).
    """
    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name", "RandomForest")
    train_all = model_cfg.get("train_all", False)
    selection_metric = model_cfg.get("selection_metric", "f1_score")
    random_state = config.get("data", {}).get("random_state", 42)

    available = _build_models(model_cfg, random_state)

    if train_all:
        models_to_train = available
    else:
        if model_name not in available:
            raise ValueError(
                f"Modelo no soportado: {model_name}. Opciones: {list(available)}"
            )
        models_to_train = {model_name: available[model_name]}

    print(f"Modelos a entrenar: {list(models_to_train)}")

    results = {}
    trained_objs = {}
    for name, model in models_to_train.items():
        print(f"Entrenando {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        }
        results[name] = metrics
        trained_objs[name] = model
        print(
            f"  {name} -> accuracy={metrics['accuracy']:.4f}, "
            f"recall={metrics['recall']:.4f}, f1={metrics['f1_score']:.4f}"
        )

    if selection_metric not in ("accuracy", "recall", "f1_score"):
        selection_metric = "f1_score"

    best_name = max(results, key=lambda n: results[n][selection_metric])
    best_metrics = results[best_name]
    best_model = trained_objs[best_name]

    paths = config.get("paths", {})
    model_path = paths.get("model_save", "models/model.pkl")
    features_path = paths.get("features_save", "models/features.pkl")

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(features_path) or ".", exist_ok=True)
    joblib.dump(best_model, model_path)
    joblib.dump(list(X_train.columns), features_path)

    print(f"\nMejor modelo: {best_name} (por {selection_metric})")
    print(f"Modelo guardado en: {model_path}")
    print(f"Features guardadas en: {features_path}")

    return {
        "accuracy": best_metrics["accuracy"],
        "recall": best_metrics["recall"],
        "f1_score": best_metrics["f1_score"],
        "best_model": best_name,
        "all_metrics": results,
    }
