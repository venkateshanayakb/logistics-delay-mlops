"""
Model Training Pipeline
========================
Train classifiers with RandomizedSearchCV screening + BayesSearchCV
fine-tuning, and Weights & Biases experiment tracking.

Strategy:
    Phase 1 — RandomizedSearchCV (fast screening) on RF + LightGBM
    Phase 2 — BayesSearchCV (fine-tuning) on the winner only

Usage:
    python -m src.train                    # Full training run
    python -m src.train --quick            # Quick smoke test
    python -m src.train --csv              # Load from CSV instead of Postgres
"""

import os
import time
import argparse
import logging
import warnings

import joblib
import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from scipy.stats import randint, uniform
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

from src.preprocessing import prepare_data

# ── Setup ────────────────────────────────────────────────────────
load_dotenv()
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

LABEL_MAP = {-1: "Early", 0: "On-time", 1: "Late"}


# ── Phase 1: RandomizedSearchCV configs (fast screening) ─────────
def get_random_search_configs(quick: bool = False):
    """
    Return {name: (estimator, param_distributions)} for RandomizedSearchCV.
    Uses scipy distributions for continuous/integer sampling.
    """
    n_iter = 5 if quick else 15

    configs = {
        "RandomForest": {
            "estimator": RandomForestClassifier(
                random_state=42, n_jobs=-1, class_weight="balanced",
            ),
            "param_distributions": {
                "n_estimators": randint(200, 600),
                "max_depth": randint(10, 30),
                "min_samples_split": randint(5, 25),
                "min_samples_leaf": randint(2, 12),
                "max_features": ["sqrt", "log2"],
            },
            "n_iter": n_iter,
        },
    }

    if HAS_LGBM:
        configs["LightGBM"] = {
            "estimator": LGBMClassifier(
                random_state=42, n_jobs=-1, verbose=-1,
                class_weight="balanced",
            ),
            "param_distributions": {
                "n_estimators": randint(200, 600),
                "num_leaves": randint(31, 127),
                "learning_rate": uniform(0.01, 0.19),  # 0.01 to 0.20
                "subsample": uniform(0.7, 0.25),  # 0.70 to 0.95
                "colsample_bytree": uniform(0.6, 0.35),  # 0.60 to 0.95
                "min_child_samples": randint(10, 50),
                "reg_alpha": uniform(0.01, 5.0),
                "reg_lambda": uniform(0.01, 5.0),
            },
            "n_iter": n_iter,
        }
    else:
        logger.warning("⚠️ LightGBM not installed — training RandomForest only")

    return configs


# ── Phase 2: BayesSearchCV config (fine-tune winner) ─────────────
def get_bayes_config(winner_name: str, winner_model, quick: bool = False):
    """
    Return BayesSearchCV search space for the winning model.
    Narrow ranges around the winner's best params for fine-tuning.
    """
    n_iter = 3 if quick else 5
    params = winner_model.get_params()

    if winner_name == "RandomForest":
        search_space = {
            "n_estimators": Integer(
                max(100, params["n_estimators"] - 100),
                min(800, params["n_estimators"] + 200),
            ),
            "max_depth": Integer(
                max(5, params["max_depth"] - 5),
                min(40, params["max_depth"] + 5),
            ),
            "min_samples_split": Integer(
                max(2, params["min_samples_split"] - 3),
                min(30, params["min_samples_split"] + 5),
            ),
            "min_samples_leaf": Integer(
                max(1, params["min_samples_leaf"] - 2),
                min(15, params["min_samples_leaf"] + 3),
            ),
            "max_features": Categorical(["sqrt", "log2"]),
        }
        estimator = RandomForestClassifier(
            random_state=42, n_jobs=-1, class_weight="balanced",
        )
    else:  # LightGBM
        search_space = {
            "n_estimators": Integer(
                max(100, params["n_estimators"] - 100),
                min(800, params["n_estimators"] + 200),
            ),
            "num_leaves": Integer(
                max(15, params["num_leaves"] - 20),
                min(200, params["num_leaves"] + 30),
            ),
            "learning_rate": Real(
                max(0.005, params["learning_rate"] * 0.5),
                min(0.3, params["learning_rate"] * 2.0),
                prior="log-uniform",
            ),
            "subsample": Real(
                max(0.5, params["subsample"] - 0.1),
                min(1.0, params["subsample"] + 0.1),
            ),
            "colsample_bytree": Real(
                max(0.4, params["colsample_bytree"] - 0.1),
                min(1.0, params["colsample_bytree"] + 0.1),
            ),
            "min_child_samples": Integer(
                max(5, params["min_child_samples"] - 10),
                min(80, params["min_child_samples"] + 15),
            ),
        }
        estimator = LGBMClassifier(
            random_state=42, n_jobs=-1, verbose=-1,
            class_weight="balanced",
        )

    return {
        "estimator": estimator,
        "search_space": search_space,
        "n_iter": n_iter,
    }


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

    if y_proba is not None:
        try:
            metrics["roc_auc_ovr"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro"
            )
        except ValueError:
            metrics["roc_auc_ovr"] = None

    return metrics


# ── Full Training Pipeline ──────────────────────────────────────
def run_training(quick: bool = False, from_csv: bool = False, use_wandb: bool = True) -> dict:
    """
    Training pipeline:
        1. Load & preprocess data
        2. Phase 1: RandomizedSearchCV screening (RF + LightGBM)
        3. Phase 2: BayesSearchCV fine-tuning (winner only)
        4. Save pipeline as joblib + log to W&B
    """
    # ── 1. Prepare data ──────────────────────────────────────────
    logger.info("Step 1/6: Loading and preprocessing data...")
    data = prepare_data(from_csv=from_csv)

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    preprocessor = data["preprocessor"]

    # Encode labels (LightGBM requires 0-indexed)
    label_encoder = LabelEncoder()
    label_encoder.fit(sorted(y_train.unique()))  # [-1, 0, 1] → [0, 1, 2]
    y_train_enc = pd.Series(
        label_encoder.transform(y_train), index=data["y_train"].index
    )
    y_test_enc = pd.Series(
        label_encoder.transform(y_test), index=data["y_test"].index
    )
    mapping = dict(zip(
        label_encoder.classes_,
        label_encoder.transform(label_encoder.classes_),
    ))
    logger.info(f"  Label mapping: {mapping}")

    # ── 2. Fit preprocessor & transform ──────────────────────────
    logger.info("Step 2/6: Fitting preprocessor...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    logger.info(f"  Train shape: {X_train_processed.shape} | Test shape: {X_test_processed.shape}")

    # No SMOTE — using class_weight="balanced" in classifiers instead
    logger.info("  ℹ️  Using class_weight='balanced' (no SMOTE)")

    # ── 3. Init W&B (optional) ────────────────────────────────────
    run = None
    if use_wandb:
        try:
            logger.info("Step 3/6: Initializing W&B run...")
            run = wandb.init(
                project=os.getenv("WANDB_PROJECT", "logistics-delay-mlops"),
                entity=os.getenv("WANDB_ENTITY"),
                name=f"train-{'quick' if quick else 'full'}-{time.strftime('%Y%m%d-%H%M%S')}",
                config={
                    "quick_mode": quick,
                    "train_samples": X_train_processed.shape[0],
                    "test_samples": X_test_processed.shape[0],
                    "n_features": X_train_processed.shape[1],
                    "smote_applied": False,
                    "strategy": "RandomizedSearch + BayesSearch fine-tune",
                    "numeric_features": data["numeric_cols"],
                    "categorical_features": data["categorical_cols"],
                },
                tags=["training", "quick" if quick else "full"],
            )
        except Exception as e:
            logger.warning(f"⚠️ W&B init failed: {e}")
            logger.warning("  Continuing without W&B logging.")
            run = None
    else:
        logger.info("Step 3/6: Skipping W&B (--no-wandb flag)")

    # ── 4. Phase 1: RandomizedSearchCV screening ─────────────────
    logger.info("Step 4/6: Phase 1 — RandomizedSearchCV screening...")
    configs = get_random_search_configs(quick=quick)
    phase1_results = []

    for name, config in configs.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"  Screening: {name}")
        logger.info(f"{'='*60}")

        start = time.time()
        search = RandomizedSearchCV(
            estimator=config["estimator"],
            param_distributions=config["param_distributions"],
            n_iter=config["n_iter"],
            cv=3,
            scoring="f1_macro",
            n_jobs=-1,
            random_state=42,
            verbose=0,
        )
        search.fit(X_train_processed, y_train_enc)
        elapsed = time.time() - start

        model = search.best_estimator_
        y_pred = model.predict(X_test_processed)
        y_proba = model.predict_proba(X_test_processed) if hasattr(model, "predict_proba") else None
        metrics = compute_metrics(y_test_enc.values, y_pred, y_proba)

        logger.info(f"  CV F1 (macro):  {search.best_score_:.4f}")
        logger.info(f"  Test F1 (macro): {metrics['f1_macro']:.4f}")
        logger.info(f"  Test Accuracy:   {metrics['accuracy']:.4f}")
        logger.info(f"  Time: {elapsed:.1f}s")

        # ── Log per-iteration CV results to W&B (creates visible curves) ──
        if wandb.run is not None:
            cv_results = search.cv_results_
            for i in range(len(cv_results["mean_test_score"])):
                wandb.log({
                    f"{name}/cv_mean_f1": cv_results["mean_test_score"][i],
                    f"{name}/cv_std_f1": cv_results["std_test_score"][i],
                    f"{name}/cv_rank": int(cv_results["rank_test_score"][i]),
                    f"{name}/iteration": i + 1,
                })
            # Also log the final best metrics
            wandb.log({
                f"{name}/phase1_best_cv_f1": search.best_score_,
                f"{name}/phase1_test_f1": metrics["f1_macro"],
                f"{name}/phase1_test_accuracy": metrics["accuracy"],
                f"{name}/phase1_test_precision": metrics["precision_macro"],
                f"{name}/phase1_test_recall": metrics["recall_macro"],
                f"{name}/phase1_time": elapsed,
            })
            if metrics.get("roc_auc_ovr") is not None:
                wandb.log({f"{name}/phase1_roc_auc": metrics["roc_auc_ovr"]})

        phase1_results.append({
            "name": name,
            "model": model,
            "best_params": search.best_params_,
            "cv_score": search.best_score_,
            "test_metrics": metrics,
            "train_time": elapsed,
            "y_pred": y_pred,
            "y_proba": y_proba,
        })

    # Pick Phase 1 winner
    winner = max(phase1_results, key=lambda r: r["test_metrics"]["f1_macro"])
    logger.info(f"\n  ⭐ Phase 1 winner: {winner['name']} (F1: {winner['test_metrics']['f1_macro']:.4f})")

    # ── 5. Phase 2: BayesSearchCV fine-tuning (winner only) ──────
    logger.info("Step 5/6: Phase 2 — BayesSearchCV fine-tuning...")
    bayes_config = get_bayes_config(winner["name"], winner["model"], quick=quick)

    start = time.time()
    bayes_search = BayesSearchCV(
        estimator=bayes_config["estimator"],
        search_spaces=bayes_config["search_space"],
        n_iter=bayes_config["n_iter"],
        cv=3,
        scoring="f1_macro",
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    bayes_search.fit(X_train_processed, y_train_enc)
    bayes_time = time.time() - start

    final_model = bayes_search.best_estimator_
    y_pred_final = final_model.predict(X_test_processed)
    y_proba_final = final_model.predict_proba(X_test_processed) if hasattr(final_model, "predict_proba") else None
    final_metrics = compute_metrics(y_test_enc.values, y_pred_final, y_proba_final)

    logger.info(f"  BayesSearch CV F1: {bayes_search.best_score_:.4f}")
    logger.info(f"  Test F1 (macro):   {final_metrics['f1_macro']:.4f}")
    logger.info(f"  Test Accuracy:     {final_metrics['accuracy']:.4f}")
    logger.info(f"  Fine-tune time:    {bayes_time:.1f}s")

    # ── Log per-iteration BayesSearch results to W&B ──
    if wandb.run is not None:
        bayes_cv = bayes_search.cv_results_
        for i in range(len(bayes_cv["mean_test_score"])):
            wandb.log({
                f"{winner['name']}/bayes_cv_f1_iter": bayes_cv["mean_test_score"][i],
                f"{winner['name']}/bayes_std_f1_iter": bayes_cv["std_test_score"][i],
                f"{winner['name']}/bayes_iteration": i + 1,
            })
        wandb.log({
            f"{winner['name']}/bayes_best_cv_f1": bayes_search.best_score_,
            f"{winner['name']}/bayes_test_f1": final_metrics["f1_macro"],
            f"{winner['name']}/bayes_test_accuracy": final_metrics["accuracy"],
            f"{winner['name']}/bayes_time": bayes_time,
        })

    # Use fine-tuned model if it improved, otherwise keep phase1 winner
    if final_metrics["f1_macro"] >= winner["test_metrics"]["f1_macro"]:
        logger.info("  ✅ Fine-tuning improved metrics — using BayesSearch model")
        best_result = {
            "name": winner["name"],
            "model": final_model,
            "best_params": bayes_search.best_params_,
            "cv_score": bayes_search.best_score_,
            "test_metrics": final_metrics,
            "train_time": winner["train_time"] + bayes_time,
            "y_pred": y_pred_final,
            "y_proba": y_proba_final,
        }
    else:
        logger.info("  ℹ️  Fine-tuning did not improve — keeping Phase 1 model")
        best_result = winner

    # ── 6. Save & report ─────────────────────────────────────────
    logger.info("Step 6/6: Saving best model...")
    logger.info(f"\n🏆 Best Model: {best_result['name']}")
    logger.info(f"   F1 (macro): {best_result['test_metrics']['f1_macro']:.4f}")
    logger.info(f"   Accuracy:   {best_result['test_metrics']['accuracy']:.4f}")

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
        "label_encoder": label_encoder,
    }

    model_path = os.path.join(MODEL_DIR, "best_pipeline.joblib")
    joblib.dump(pipeline_artifact, model_path)
    logger.info(f"  💾 Saved pipeline to: {model_path}")

    # Classification report
    y_test_decoded = label_encoder.inverse_transform(y_test_enc)
    y_pred_decoded = label_encoder.inverse_transform(best_result["y_pred"])
    report = classification_report(
        y_test_decoded, y_pred_decoded,
        target_names=[LABEL_MAP[k] for k in sorted(LABEL_MAP.keys())],
    )
    logger.info(f"\n{report}")

    # Log to W&B — rich visualizations
    if run is not None:
        class_names = [LABEL_MAP[k] for k in sorted(LABEL_MAP.keys())]

        try:
            # ── Confusion Matrix heatmap ──
            y_test_int = list(y_test_enc.values.astype(int))
            y_pred_int = list(best_result["y_pred"].astype(int))
            wandb.log({
                "confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_test_int,
                    preds=y_pred_int,
                    class_names=class_names,
                ),
            })
        except Exception as e:
            logger.warning(f"⚠️ Confusion matrix logging failed: {e}")

        try:
            # ── Per-class metrics bar chart (as a W&B Table) ──
            y_test_labels = [LABEL_MAP[v] for v in label_encoder.inverse_transform(y_test_enc)]
            y_pred_labels = [LABEL_MAP[v] for v in label_encoder.inverse_transform(best_result["y_pred"])]
            per_class_report = classification_report(
                y_test_labels, y_pred_labels,
                target_names=class_names, output_dict=True,
            )
            metrics_table = wandb.Table(
                columns=["Class", "Precision", "Recall", "F1-Score", "Support"],
            )
            for cls in class_names:
                r = per_class_report[cls]
                metrics_table.add_data(cls, r["precision"], r["recall"], r["f1-score"], r["support"])
            wandb.log({
                "per_class_metrics": metrics_table,
                "per_class_f1_bar": wandb.plot.bar(
                    wandb.Table(
                        data=[[cls, per_class_report[cls]["f1-score"]] for cls in class_names],
                        columns=["Class", "F1-Score"],
                    ),
                    "Class", "F1-Score", title="F1-Score by Class",
                ),
                "per_class_precision_bar": wandb.plot.bar(
                    wandb.Table(
                        data=[[cls, per_class_report[cls]["precision"]] for cls in class_names],
                        columns=["Class", "Precision"],
                    ),
                    "Class", "Precision", title="Precision by Class",
                ),
                "per_class_recall_bar": wandb.plot.bar(
                    wandb.Table(
                        data=[[cls, per_class_report[cls]["recall"]] for cls in class_names],
                        columns=["Class", "Recall"],
                    ),
                    "Class", "Recall", title="Recall by Class",
                ),
            })
        except Exception as e:
            logger.warning(f"⚠️ Per-class metrics logging failed: {e}")

        try:
            # ── Model comparison table ──
            comparison_table = wandb.Table(
                columns=["Model", "F1 (macro)", "Accuracy", "Precision", "Recall", "ROC-AUC", "Time (s)"],
            )
            for r in phase1_results:
                m = r["test_metrics"]
                comparison_table.add_data(
                    r["name"], m["f1_macro"], m["accuracy"],
                    m["precision_macro"], m["recall_macro"],
                    m.get("roc_auc_ovr", 0), round(r["train_time"], 1),
                )
            wandb.log({
                "model_comparison": comparison_table,
                "model_f1_comparison": wandb.plot.bar(
                    wandb.Table(
                        data=[[r["name"], r["test_metrics"]["f1_macro"]] for r in phase1_results],
                        columns=["Model", "F1 (macro)"],
                    ),
                    "Model", "F1 (macro)", title="Model F1 Comparison",
                ),
            })
        except Exception as e:
            logger.warning(f"⚠️ Model comparison logging failed: {e}")

        # ── Summary metrics ──
        wandb.log({
            "best_model": best_result["name"],
            "best_f1_macro": best_result["test_metrics"]["f1_macro"],
            "best_accuracy": best_result["test_metrics"]["accuracy"],
        })

        # ── Artifact ──
        artifact = wandb.Artifact(
            name="best-pipeline", type="model",
            description=f"Best: {best_result['name']} | F1: {best_result['test_metrics']['f1_macro']:.4f}",
        )
        artifact.add_file(model_path)
        run.log_artifact(artifact)
        wandb.log({"classification_report": wandb.Html(f"<pre>{report}</pre>")})
        wandb.finish()
        logger.info("✅ Training complete! W&B run finished.")
    else:
        logger.info("✅ Training complete! (W&B logging skipped)")

    all_results = phase1_results + [best_result]
    return {
        "best_result": best_result,
        "all_results": all_results,
        "pipeline_path": model_path,
        "data": data,
    }


# ── CLI ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Pipeline")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test")
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
        name = r['name']
        f1 = r['test_metrics']['f1_macro']
        acc = r['test_metrics']['accuracy']
        t = r['train_time']
        print(f"  {name:20s} | F1: {f1:.4f} | Acc: {acc:.4f} | Time: {t:.1f}s")
