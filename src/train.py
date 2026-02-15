"""
Model Training Pipeline
========================
Train multiple classifiers with Bayesian hyperparameter search,
SMOTE oversampling, and full Weights & Biases experiment tracking.

Usage:
    python -m src.train                    # Full training run
    python -m src.train --quick            # Quick smoke test (fewer iterations)
    python -m src.train --csv              # Load from CSV instead of Postgres
"""

import os
import time
import argparse
import logging
import warnings

import joblib
import numpy as np
import wandb
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real

from src.preprocessing import prepare_data

# ── Setup ────────────────────────────────────────────────────────
load_dotenv()
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

LABEL_MAP = {-1: "Early", 0: "On-time", 1: "Late"}


# ── Classifier Configurations ────────────────────────────────────
def get_classifier_configs(quick: bool = False):
    """
    Return a dict of {name: (estimator, search_space)} pairs.
    If quick=True, reduce search iterations for fast smoke testing.
    """
    n_iter = 5 if quick else 30

    configs = {
        "RandomForest": {
            "estimator": RandomForestClassifier(random_state=42, n_jobs=-1),
            "search_space": {
                "n_estimators": Integer(100, 500),
                "max_depth": Integer(5, 30),
                "min_samples_split": Integer(2, 20),
                "min_samples_leaf": Integer(1, 10),
                "max_features": Categorical(["sqrt", "log2"]),
            },
            "n_iter": n_iter,
        },
        "GradientBoosting": {
            "estimator": GradientBoostingClassifier(random_state=42),
            "search_space": {
                "n_estimators": Integer(100, 400),
                "max_depth": Integer(3, 15),
                "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
                "subsample": Real(0.6, 1.0),
                "min_samples_split": Integer(2, 20),
            },
            "n_iter": n_iter,
        },
    }

    return configs


# ── Compute Metrics ──────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Compute classification metrics for multi-class."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
    }

    # ROC-AUC (one-vs-rest) — needs probability estimates
    if y_proba is not None:
        try:
            metrics["roc_auc_ovr"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro"
            )
        except ValueError:
            metrics["roc_auc_ovr"] = None

    return metrics


# ── Train Single Classifier ─────────────────────────────────────
def train_classifier(
    name: str,
    config: dict,
    X_train_processed: np.ndarray,
    y_train_resampled: np.ndarray,
    X_test_processed: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Train one classifier with BayesSearchCV and log to W&B.
    Returns dict with model, best_params, and metrics.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"  Training: {name}")
    logger.info(f"{'='*60}")

    start_time = time.time()

    # Bayesian hyperparameter search with cross-validation
    bayes_search = BayesSearchCV(
        estimator=config["estimator"],
        search_spaces=config["search_space"],
        n_iter=config["n_iter"],
        cv=3,
        scoring="f1_macro",
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )

    bayes_search.fit(X_train_processed, y_train_resampled)
    train_time = time.time() - start_time

    best_model = bayes_search.best_estimator_
    best_params = bayes_search.best_params_
    cv_score = bayes_search.best_score_

    # Predictions on test set
    y_pred = best_model.predict(X_test_processed)
    y_proba = (
        best_model.predict_proba(X_test_processed)
        if hasattr(best_model, "predict_proba")
        else None
    )

    # Compute metrics
    test_metrics = compute_metrics(y_test, y_pred, y_proba)

    logger.info(f"  Best CV F1 (macro): {cv_score:.4f}")
    logger.info(f"  Test Accuracy:      {test_metrics['accuracy']:.4f}")
    logger.info(f"  Test F1 (macro):    {test_metrics['f1_macro']:.4f}")
    logger.info(f"  Train time:         {train_time:.1f}s")

    # Log to W&B (if active)
    if wandb.run is not None:
        wandb.log({
            f"{name}/cv_f1_macro": cv_score,
            f"{name}/test_accuracy": test_metrics["accuracy"],
            f"{name}/test_f1_macro": test_metrics["f1_macro"],
            f"{name}/test_f1_weighted": test_metrics["f1_weighted"],
            f"{name}/test_precision_macro": test_metrics["precision_macro"],
            f"{name}/test_recall_macro": test_metrics["recall_macro"],
            f"{name}/test_roc_auc_ovr": test_metrics.get("roc_auc_ovr"),
            f"{name}/train_time_seconds": train_time,
            f"{name}/best_params": best_params,
        })

    return {
        "name": name,
        "model": best_model,
        "best_params": best_params,
        "cv_score": cv_score,
        "test_metrics": test_metrics,
        "train_time": train_time,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


# ── Full Training Pipeline ──────────────────────────────────────
def run_training(quick: bool = False, from_csv: bool = False, use_wandb: bool = True) -> dict:
    """
    Full training pipeline:
        1. Load & preprocess data
        2. Apply SMOTE oversampling
        3. Train multiple classifiers with BayesSearchCV
        4. Select best model
        5. Save pipeline (preprocessor + model) as joblib
        6. Log everything to W&B

    Returns:
        dict with best model info and all results.
    """
    # ── 1. Prepare data ──────────────────────────────────────────
    logger.info("Step 1/6: Loading and preprocessing data...")
    data = prepare_data(from_csv=from_csv)

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    preprocessor = data["preprocessor"]

    # ── 2. Fit preprocessor & transform ──────────────────────────
    logger.info("Step 2/6: Fitting preprocessor...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    logger.info(f"  Transformed train shape: {X_train_processed.shape}")
    logger.info(f"  Transformed test shape:  {X_test_processed.shape}")

    # ── 3. SMOTE oversampling ────────────────────────────────────
    logger.info("Step 3/6: Applying SMOTE oversampling...")
    logger.info(f"  Before SMOTE: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

    logger.info(f"  After SMOTE:  {dict(zip(*np.unique(y_train_resampled, return_counts=True)))}")

    # ── 4. Init W&B (optional) ────────────────────────────────────
    run = None
    if use_wandb:
        try:
            logger.info("Step 4/6: Initializing W&B run...")
            run = wandb.init(
                project=os.getenv("WANDB_PROJECT", "logistics-delay-mlops"),
                entity=os.getenv("WANDB_ENTITY"),
                name=f"train-{'quick' if quick else 'full'}-{time.strftime('%Y%m%d-%H%M%S')}",
                config={
                    "quick_mode": quick,
                    "train_samples": X_train_resampled.shape[0],
                    "test_samples": X_test_processed.shape[0],
                    "n_features": X_train_processed.shape[1],
                    "smote_applied": True,
                    "numeric_features": data["numeric_cols"],
                    "categorical_features": data["categorical_cols"],
                },
                tags=["training", "quick" if quick else "full"],
            )
        except Exception as e:
            logger.warning(f"⚠️ W&B init failed: {e}")
            logger.warning("  Continuing training without W&B logging.")
            run = None
    else:
        logger.info("Step 4/6: Skipping W&B (--no-wandb flag)")

    # ── 5. Train all classifiers ─────────────────────────────────
    logger.info("Step 5/6: Training classifiers...")
    configs = get_classifier_configs(quick=quick)
    results = []

    for name, config in configs.items():
        result = train_classifier(
            name, config,
            X_train_resampled, y_train_resampled,
            X_test_processed, y_test.values,
        )
        results.append(result)

    # ── 6. Select best model & save ──────────────────────────────
    logger.info("Step 6/6: Selecting best model and saving...")
    best_result = max(results, key=lambda r: r["test_metrics"]["f1_macro"])

    logger.info(f"\n🏆 Best Model: {best_result['name']}")
    logger.info(f"   F1 (macro): {best_result['test_metrics']['f1_macro']:.4f}")
    logger.info(f"   Accuracy:   {best_result['test_metrics']['accuracy']:.4f}")

    # Save complete pipeline: preprocessor + best model
    pipeline_artifact = {
        "preprocessor": preprocessor,
        "model": best_result["model"],
        "model_name": best_result["name"],
        "best_params": best_result["best_params"],
        "test_metrics": best_result["test_metrics"],
        "feature_names": data["feature_names"],
        "numeric_cols": data["numeric_cols"],
        "categorical_cols": data["categorical_cols"],
        "label_map": LABEL_MAP,
    }

    model_path = os.path.join(MODEL_DIR, "best_pipeline.joblib")
    joblib.dump(pipeline_artifact, model_path)
    logger.info(f"  💾 Saved pipeline to: {model_path}")

    # Classification report
    report = classification_report(
        y_test, best_result["y_pred"],
        target_names=[LABEL_MAP[k] for k in sorted(LABEL_MAP.keys())],
    )
    logger.info(f"\n{report}")

    # Log to W&B (if active)
    if run is not None:
        wandb.log({
            "best_model": best_result["name"],
            "best_f1_macro": best_result["test_metrics"]["f1_macro"],
            "best_accuracy": best_result["test_metrics"]["accuracy"],
        })

        artifact = wandb.Artifact(
            name="best-pipeline",
            type="model",
            description=f"Best model: {best_result['name']} | F1: {best_result['test_metrics']['f1_macro']:.4f}",
        )
        artifact.add_file(model_path)
        run.log_artifact(artifact)
        wandb.log({"classification_report": wandb.Html(f"<pre>{report}</pre>")})
        wandb.finish()
        logger.info("✅ Training complete! W&B run finished.")
    else:
        logger.info("✅ Training complete! (W&B logging skipped)")

    return {
        "best_result": best_result,
        "all_results": results,
        "pipeline_path": model_path,
        "data": data,
    }


# ── CLI ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Pipeline")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test (5 iterations)")
    parser.add_argument("--csv", action="store_true", help="Load from CSV instead of Postgres")
    parser.add_argument("--no-wandb", action="store_true", help="Skip W&B logging")
    args = parser.parse_args()

    result = run_training(quick=args.quick, from_csv=args.csv, use_wandb=not args.no_wandb)

    print(f"\n{'='*60}")
    print(f"  🏆 Best Model: {result['best_result']['name']}")
    print(f"  📊 F1 (macro): {result['best_result']['test_metrics']['f1_macro']:.4f}")
    print(f"  📊 Accuracy:   {result['best_result']['test_metrics']['accuracy']:.4f}")
    print(f"  💾 Saved to:   {result['pipeline_path']}")
    print(f"{'='*60}")

    print("\n── All Results ──")
    for r in result["all_results"]:
        print(f"  {r['name']:20s} | F1: {r['test_metrics']['f1_macro']:.4f} | Acc: {r['test_metrics']['accuracy']:.4f} | Time: {r['train_time']:.1f}s")
