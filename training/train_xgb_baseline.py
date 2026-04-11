import os
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from feature_repo.feature_definitions import LABEL_COLUMN, MODEL_FEATURES
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("DATA_DIR", REPO_ROOT / "data/processed/dataset_retailrocket"))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "xgb-baseline-retailrocket")


def load_split(name: str):
    split_path = DATA_DIR / f"{name}.parquet"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Missing split file: {split_path}. "
            "Run data/scripts/preprocess_retailrocket.py and "
            "data/scripts/build_trainset_retailrocket.py first."
        )
    df = pd.read_parquet(split_path)
    missing_features = [feature for feature in MODEL_FEATURES if feature not in df.columns]
    if missing_features:
        raise ValueError(
            f"Split file {split_path} is missing canonical feature columns from feature_repo: {missing_features}"
        )
    if LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"Split file {split_path} is missing canonical label column '{LABEL_COLUMN}'"
        )
    X = df[MODEL_FEATURES].copy()
    y = df[LABEL_COLUMN].astype(int).copy()
    return X, y


def eval_metrics(y_true, proba, prefix: str):
    eps = 1e-15
    proba = np.clip(proba, eps, 1 - eps)
    return {
        f"{prefix}_roc_auc": roc_auc_score(y_true, proba),
        f"{prefix}_pr_auc": average_precision_score(y_true, proba),
        f"{prefix}_logloss": log_loss(y_true, proba),
    }


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    scale_pos_weight = max((y_train == 0).sum() / max((y_train == 1).sum(), 1), 1.0)

    with mlflow.start_run() as run:
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
        )

        model.fit(X_train, y_train)

        val_proba = model.predict_proba(X_val)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]

        val_metrics = eval_metrics(y_val, val_proba, "val")
        test_metrics = eval_metrics(y_test, test_proba, "test")

        mlflow.log_params(model.get_params())
        mlflow.log_metrics(val_metrics)
        mlflow.log_metrics(test_metrics)
        mlflow.xgboost.log_model(model, artifact_path="model")

        print("Run ID:", run.info.run_id)
        print("Model URI:", f"runs:/{run.info.run_id}/model")
        print("Val metrics:", val_metrics)
        print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()

