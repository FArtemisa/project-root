import os
import pytest

from src.main import load_config
from src.data_loader import load_and_preprocess_data
from src.model_trainer import train_and_save_model


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def data(config):
    return load_and_preprocess_data(config)


def test_load_and_preprocess_data_not_empty(data):
    X_train, X_test, y_train, y_test = data
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0
    assert X_train.shape[1] == X_test.shape[1]


def test_train_and_save_model_returns_metrics(config, data):
    X_train, X_test, y_train, y_test = data
    cfg = dict(config)
    cfg["model"] = dict(cfg["model"])
    cfg["model"]["train_all"] = False
    cfg["model"]["name"] = "LogisticRegression"

    metrics = train_and_save_model(X_train, y_train, X_test, y_test, cfg)

    for key in ("accuracy", "recall", "f1_score"):
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0

    assert os.path.exists(cfg["paths"]["model_save"])
