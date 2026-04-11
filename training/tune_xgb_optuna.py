import argparse
import os
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
from feature_repo.feature_definitions import LABEL_COLUMN, MODEL_FEATURES
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("DATA_DIR", REPO_ROOT / "data/processed/dataset_retailrocket"))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune XGBoost with Optuna and MLflow logging.")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of Optuna trials.")
    parser.add_argument("--timeout-sec", type=int, default=None, help="Optional timeout in seconds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        help="MLflow tracking URI.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=os.getenv("MLFLOW_EXPERIMENT", "xgb-optuna-retailrocket"),
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="xgb-optuna-retailrocket",
        help="Optuna study name (for logging only).",
    )
    parser.add_argument(
        "--objective-metric",
        type=str,
        default="val_pr_auc",
        choices=["val_pr_auc", "val_roc_auc", "val_logloss"],
        help="Metric used by Optuna objective.",
    )
    return parser.parse_args()


def load_split(name: str) -> Tuple[pd.DataFrame, pd.Series]:
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


def eval_metrics(y_true: pd.Series, proba: np.ndarray, prefix: str) -> Dict[str, float]:
    eps = 1e-15
    proba = np.clip(proba, eps, 1 - eps)
    return {
        f"{prefix}_roc_auc": float(roc_auc_score(y_true, proba)),
        f"{prefix}_pr_auc": float(average_precision_score(y_true, proba)),
        f"{prefix}_logloss": float(log_loss(y_true, proba)),
    }


def train_and_evaluate(
    params: Dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int,
) -> Tuple[XGBClassifier, Dict[str, float], Dict[str, float]]:
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        **params,
    )
    model.fit(X_train, y_train)

    val_proba = model.predict_proba(X_val)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]
    return model, eval_metrics(y_val, val_proba, "val"), eval_metrics(y_test, test_proba, "test")


def main() -> None:
    args = parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    X_train_val = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_train_val = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

    scale_pos_weight = max((y_train == 0).sum() / max((y_train == 1).sum(), 1), 1.0)

    baseline_params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
    }

    with mlflow.start_run(run_name=f"{args.study_name}-summary") as parent_run:
        mlflow.set_tags(
            {
                "pipeline": "optuna_tuning",
                "study_name": args.study_name,
                "objective_metric": args.objective_metric,
            }
        )
        mlflow.log_params(
            {
                "n_trials": args.n_trials,
                "timeout_sec": args.timeout_sec if args.timeout_sec is not None else -1,
                "seed": args.seed,
                "scale_pos_weight": scale_pos_weight,
            }
        )

        baseline_model, baseline_val, baseline_test = train_and_evaluate(
            baseline_params, X_train, y_train, X_val, y_val, X_test, y_test, args.seed
        )
        mlflow.log_params({f"baseline_{k}": v for k, v in baseline_params.items()})
        mlflow.log_metrics({f"baseline_{k}": v for k, v in baseline_val.items()})
        mlflow.log_metrics({f"baseline_{k}": v for k, v in baseline_test.items()})

        sampler = optuna.samplers.TPESampler(seed=args.seed)
        direction = "minimize" if args.objective_metric == "val_logloss" else "maximize"
        study = optuna.create_study(direction=direction, sampler=sampler)

        def objective(trial: optuna.Trial) -> float:
            trial_params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-2, 3e-1, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "scale_pos_weight": scale_pos_weight,
            }
            with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                mlflow.log_param("trial_number", trial.number)
                mlflow.log_params(trial_params)
                _, val_metrics, _ = train_and_evaluate(
                    trial_params, X_train, y_train, X_val, y_val, X_test, y_test, args.seed
                )
                mlflow.log_metrics(val_metrics)
                objective_value = val_metrics[args.objective_metric]
                mlflow.log_metric("objective_value", objective_value)
                return objective_value

        study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout_sec)

        best_params = dict(study.best_params)
        best_params["scale_pos_weight"] = scale_pos_weight

        tuned_model, tuned_val, tuned_test = train_and_evaluate(
            best_params, X_train, y_train, X_val, y_val, X_test, y_test, args.seed
        )
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metrics({f"tuned_{k}": v for k, v in tuned_val.items()})
        mlflow.log_metrics({f"tuned_{k}": v for k, v in tuned_test.items()})

        mlflow.log_metrics(
            {
                "test_roc_auc_gain": tuned_test["test_roc_auc"] - baseline_test["test_roc_auc"],
                "test_pr_auc_gain": tuned_test["test_pr_auc"] - baseline_test["test_pr_auc"],
                "test_logloss_reduction": baseline_test["test_logloss"] - tuned_test["test_logloss"],
                "best_trial_value": study.best_value,
                "best_trial_number": float(study.best_trial.number),
            }
        )

        final_model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=args.seed,
            n_jobs=-1,
            tree_method="hist",
            **best_params,
        )
        final_model.fit(X_train_val, y_train_val)
        final_test_proba = final_model.predict_proba(X_test)[:, 1]
        final_test = eval_metrics(y_test, final_test_proba, "final_test")
        mlflow.log_metrics(final_test)
        # Log a guaranteed run artifact directory containing MLmodel for serving compatibility.
        with tempfile.TemporaryDirectory() as tmp_dir:
            mlflow.xgboost.save_model(final_model, path=tmp_dir)
            mlflow.log_artifacts(tmp_dir, artifact_path="best_model_train_val")

        print("Parent run ID:", parent_run.info.run_id)
        print("Best trial:", study.best_trial.number)
        print("Best params:", best_params)
        print("Baseline test:", baseline_test)
        print("Tuned test:", tuned_test)
        print("Final test (train+val retrain):", final_test)


if __name__ == "__main__":
    main()

