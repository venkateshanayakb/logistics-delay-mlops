"""
Model Evaluation
=================
Load the best trained model, generate evaluation plots and SHAP analysis,
and log everything to Weights & Biases.

Usage:
    python -m src.evaluate                  # Full evaluation
    python -m src.evaluate --no-shap        # Skip SHAP (faster)
    python -m src.evaluate --csv            # Load data from CSV
"""

import os
import time
import argparse
import logging
import warnings

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for servers
import matplotlib.pyplot as plt
import wandb
from dotenv import load_dotenv
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

from src.preprocessing import prepare_data

# ── Setup ────────────────────────────────────────────────────────
load_dotenv()
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

REPORT_DIR = "report"
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_pipeline.joblib")
LABEL_MAP = {-1: "Early", 0: "On-time", 1: "Late"}
CLASSES = sorted(LABEL_MAP.keys())
CLASS_NAMES = [LABEL_MAP[c] for c in CLASSES]


def ensure_report_dir():
    os.makedirs(REPORT_DIR, exist_ok=True)


# ── Load Saved Pipeline ─────────────────────────────────────────
def load_pipeline(model_path: str = MODEL_PATH) -> dict:
    """Load the saved pipeline artifact (preprocessor + model + metadata)."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. Run `python -m src.train` first."
        )
    pipeline = joblib.load(model_path)
    logger.info(f"✅ Loaded pipeline: {pipeline['model_name']} from {model_path}")
    return pipeline


# ── Confusion Matrix Plot ────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, save_path: str) -> str:
    """Generate and save a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  📊 Saved confusion matrix: {save_path}")
    return save_path


# ── ROC Curves (One-vs-Rest) ────────────────────────────────────
def plot_roc_curves(y_true, y_proba, save_path: str) -> str:
    """Generate per-class ROC curves and save."""
    y_bin = label_binarize(y_true, classes=CLASSES)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#e74c3c", "#2ecc71", "#3498db"]

    for i, (cls, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves (One-vs-Rest)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  📊 Saved ROC curves: {save_path}")
    return save_path


# ── SHAP Analysis ────────────────────────────────────────────────
def run_shap_analysis(model, X_test_processed, feature_names: list, save_dir: str) -> list:
    """
    Generate SHAP plots: summary, bar, and waterfall for first sample.
    Returns list of saved file paths.
    """
    import shap

    saved_paths = []
    logger.info("  Computing SHAP values (this may take a minute)...")

    # Use TreeExplainer for tree-based models
    explainer = shap.TreeExplainer(model)
    # Use a subset for speed
    sample_size = min(500, X_test_processed.shape[0])
    X_sample = X_test_processed[:sample_size]
    shap_values = explainer.shap_values(X_sample)

    # ── Summary Plot (beeswarm) ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 8))
    # For multi-class, shap_values is a list of arrays
    if isinstance(shap_values, list):
        # Use class 1 (Late) for the summary — most interesting
        shap.summary_plot(
            shap_values[CLASSES.index(1)],
            X_sample,
            feature_names=feature_names,
            show=False,
            max_display=20,
        )
    else:
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=feature_names,
            show=False,
            max_display=20,
        )
    plt.title("SHAP Summary — Feature Impact on 'Late' Prediction", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "shap_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    saved_paths.append(path)
    logger.info(f"  📊 Saved SHAP summary: {path}")

    # ── Bar Plot (mean |SHAP|) ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    if isinstance(shap_values, list):
        shap.summary_plot(
            shap_values[CLASSES.index(1)],
            X_sample,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
            max_display=15,
        )
    else:
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
            max_display=15,
        )
    plt.title("SHAP Feature Importance (Mean |SHAP|)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "shap_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    saved_paths.append(path)
    logger.info(f"  📊 Saved SHAP bar plot: {path}")

    # ── Waterfall Plot (single prediction) ───────────────────────
    try:
        if isinstance(shap_values, list):
            sv = shap.Explanation(
                values=shap_values[CLASSES.index(1)][0],
                base_values=explainer.expected_value[CLASSES.index(1)],
                feature_names=feature_names,
            )
        else:
            sv = shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                feature_names=feature_names,
            )
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.plots.waterfall(sv, show=False, max_display=15)
        plt.title("SHAP Waterfall — Single Prediction Explanation", fontsize=13, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(save_dir, "shap_waterfall.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close("all")
        saved_paths.append(path)
        logger.info(f"  📊 Saved SHAP waterfall: {path}")
    except Exception as e:
        logger.warning(f"  ⚠️ Waterfall plot failed: {e}")

    return saved_paths


# ── Full Evaluation Pipeline ────────────────────────────────────
def run_evaluation(skip_shap: bool = False, from_csv: bool = False, use_wandb: bool = True) -> dict:
    """
    Full evaluation:
        1. Load saved pipeline
        2. Load & preprocess test data
        3. Generate predictions
        4. Plot confusion matrix, ROC curves
        5. Run SHAP analysis
        6. Log to W&B
    """
    ensure_report_dir()

    # ── 1. Load pipeline ─────────────────────────────────────────
    logger.info("Step 1/5: Loading saved pipeline...")
    pipeline = load_pipeline()
    preprocessor = pipeline["preprocessor"]
    model = pipeline["model"]

    # ── 2. Load test data ────────────────────────────────────────
    logger.info("Step 2/5: Loading test data...")
    data = prepare_data(from_csv=from_csv)
    X_test = data["X_test"]
    y_test = data["y_test"]

    # Transform with the fitted preprocessor from the pipeline
    X_test_processed = preprocessor.transform(X_test)

    # ── 3. Predictions ───────────────────────────────────────────
    logger.info("Step 3/5: Generating predictions...")
    y_pred = model.predict(X_test_processed)
    y_proba = (
        model.predict_proba(X_test_processed)
        if hasattr(model, "predict_proba")
        else None
    )

    # ── 4. Init W&B (optional) ──────────────────────────────────
    run = None
    if use_wandb:
        try:
            run = wandb.init(
                project=os.getenv("WANDB_PROJECT", "logistics-delay-mlops"),
                entity=os.getenv("WANDB_ENTITY"),
                name=f"eval-{pipeline['model_name']}-{time.strftime('%Y%m%d-%H%M%S')}",
                config={
                    "model_name": pipeline["model_name"],
                    "best_params": pipeline["best_params"],
                },
                tags=["evaluation"],
                job_type="evaluation",
            )
        except Exception as e:
            logger.warning(f"⚠️ W&B init failed: {e}")
            run = None
    else:
        logger.info("Skipping W&B (--no-wandb flag)")

    # ── 5. Generate reports ──────────────────────────────────────
    logger.info("Step 4/5: Generating evaluation reports...")

    # Classification report
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    logger.info(f"\n{report}")

    report_path = os.path.join(REPORT_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # Confusion matrix
    cm_path = plot_confusion_matrix(y_test, y_pred, os.path.join(REPORT_DIR, "confusion_matrix.png"))

    # ROC curves
    roc_path = None
    if y_proba is not None:
        roc_path = plot_roc_curves(y_test, y_proba, os.path.join(REPORT_DIR, "roc_curves.png"))

    # Log plots to W&B (if active)
    if run is not None:
        wandb.log({
            "confusion_matrix": wandb.Image(cm_path),
            "classification_report_text": wandb.Html(f"<pre>{report}</pre>"),
        })
        if roc_path:
            wandb.log({"roc_curves": wandb.Image(roc_path)})

    # ── SHAP ─────────────────────────────────────────────────────
    shap_paths = []
    if not skip_shap:
        logger.info("Step 5/5: Running SHAP analysis...")
        # Get feature names from the preprocessor
        try:
            ohe = preprocessor.named_transformers_["cat"].named_steps["encoder"]
            cat_feature_names = list(ohe.get_feature_names_out(pipeline["categorical_cols"]))
        except Exception:
            cat_feature_names = []

        feature_names = pipeline["numeric_cols"] + cat_feature_names

        # Ensure feature names match transformed shape
        if len(feature_names) != X_test_processed.shape[1]:
            feature_names = [f"f_{i}" for i in range(X_test_processed.shape[1])]

        shap_paths = run_shap_analysis(model, X_test_processed, feature_names, REPORT_DIR)

        for sp in shap_paths:
            name = os.path.splitext(os.path.basename(sp))[0]
            if run is not None:
                wandb.log({name: wandb.Image(sp)})
    else:
        logger.info("Step 5/5: Skipping SHAP analysis (--no-shap flag)")

    if run is not None:
        wandb.finish()
        logger.info("✅ Evaluation complete! Reports saved to report/ folder. W&B run finished.")
    else:
        logger.info("✅ Evaluation complete! Reports saved to report/ folder.")

    return {
        "report": report,
        "confusion_matrix_path": cm_path,
        "roc_curves_path": roc_path,
        "shap_paths": shap_paths,
    }


# ── CLI ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Evaluation Pipeline")
    parser.add_argument("--no-shap", action="store_true", help="Skip SHAP analysis (faster)")
    parser.add_argument("--csv", action="store_true", help="Load data from CSV")
    parser.add_argument("--no-wandb", action="store_true", help="Skip W&B logging")
    args = parser.parse_args()

    result = run_evaluation(skip_shap=args.no_shap, from_csv=args.csv, use_wandb=not args.no_wandb)

    print(f"\n{'='*60}")
    print(f"  Evaluation Complete!")
    print(f"{'='*60}")
    print(f"  📄 Classification Report: report/classification_report.txt")
    print(f"  📊 Confusion Matrix:      {result['confusion_matrix_path']}")
    if result["roc_curves_path"]:
        print(f"  📊 ROC Curves:            {result['roc_curves_path']}")
    if result["shap_paths"]:
        print(f"  📊 SHAP Plots:            {len(result['shap_paths'])} saved")
    print(f"{'='*60}")
