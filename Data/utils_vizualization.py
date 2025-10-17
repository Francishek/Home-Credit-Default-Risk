import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import numpy as np
from typing import Dict, Optional, List, Tuple, Union, Any, Callable
from .utils_EDA import (
    handle_rare_categories,
    create_missing_flags,
    collapse_building_flags,
    handle_anomalies,
)
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)
import lightgbm as lgb
import warnings
from sklearn.calibration import CalibrationDisplay
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline
import shap


def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_probs_lgbm: np.ndarray,
    y_probs_xgb: np.ndarray,
    model_names: Tuple[str, str] = ("LGBM", "XGB"),
) -> None:
    """
    Plot ROC and Precision-Recall curves for two classification models.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels (0 or 1).
    y_probs_lgbm : np.ndarray
        Predicted probabilities for the positive class from the first model (e.g., LGBM).
    y_probs_xgb : np.ndarray
        Predicted probabilities for the positive class from the second model (e.g., XGB).
    model_names : Tuple[str, str], optional
        Names of the models to display in the legends (default: ("LGBM", "XGB")).

    Returns
    -------
    None
        Displays matplotlib plots comparing ROC and Precision–Recall curves.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    fpr_lgbm, tpr_lgbm, _ = roc_curve(y_true, y_probs_lgbm)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_true, y_probs_xgb)

    roc_auc_lgbm = auc(fpr_lgbm, tpr_lgbm)
    roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

    axes[0].plot(
        fpr_lgbm, tpr_lgbm, label=f"{model_names[0]} (AUC = {roc_auc_lgbm:.3f})"
    )
    axes[0].plot(fpr_xgb, tpr_xgb, label=f"{model_names[1]} (AUC = {roc_auc_xgb:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.7)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].set_title("ROC Curve")
    axes[0].legend()
    axes[0].grid(True)

    prec_lgbm, rec_lgbm, _ = precision_recall_curve(y_true, y_probs_lgbm)
    prec_xgb, rec_xgb, _ = precision_recall_curve(y_true, y_probs_xgb)

    pr_auc_lgbm = auc(rec_lgbm, prec_lgbm)
    pr_auc_xgb = auc(rec_xgb, prec_xgb)

    axes[1].plot(
        rec_lgbm, prec_lgbm, label=f"{model_names[0]} (PR-AUC = {pr_auc_lgbm:.3f})"
    )
    axes[1].plot(
        rec_xgb, prec_xgb, label=f"{model_names[1]} (PR-AUC = {pr_auc_xgb:.3f})"
    )
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision–Recall Curve")
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle(
        f"Model Performance Comparison: {model_names[0]} vs {model_names[1]}",
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()


def plot_calibration_curves(
    models: List[Tuple[str, ClassifierMixin]],
    X: np.ndarray,
    y: np.ndarray,
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """
    Plot calibration curves for multiple classification models.

    Parameters
    ----------
    models : List[Tuple[str, ClassifierMixin]]
        A list of tuples containing (model_name, trained_model_instance).
        Example: [("LGBM", lgbm_model), ("XGB", xgb_model)].
    X : np.ndarray
        Feature matrix used for calibration plotting.
    y : np.ndarray
        Ground truth binary labels (0 or 1).
    figsize : Tuple[int, int], optional
        Size of the matplotlib figure (default: (10, 8)).

    Returns
    -------
    None
        Displays the calibration plots.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name, model in models:
        CalibrationDisplay.from_estimator(model, X, y, ax=ax, name=name)

    # Perfect calibration reference line
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    ax.set_title("Calibration Curves")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def analyze_shap_separate_plots(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    model_name: str = "",
    sample_size: int = 1000,
    background_size: int = 100,
):
    """
    SHAP analysis with separate figures for each plot and top 20 feature importance
    """

    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]
    X_processed = preprocessor.transform(X_train)

    try:
        feature_names = preprocessor.get_feature_names_out()
    except:
        feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]

    background = shap.sample(X_processed, background_size)

    if len(X_processed) > sample_size:
        sample_indices = np.random.choice(len(X_processed), sample_size, replace=False)
        X_sample = X_processed[sample_indices]
    else:
        X_sample = X_processed

    explainer = shap.TreeExplainer(
        classifier, data=background, feature_perturbation="interventional"
    )

    shap_values = explainer.shap_values(X_sample, check_additivity=False)

    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values_matrix = shap_values[1]
    else:
        shap_values_matrix = shap_values

    shap_df = pd.DataFrame(shap_values_matrix, columns=feature_names)

    # Compute top 15 features by mean absolute SHAP value
    mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
    top_features = mean_abs_shap.head(20)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values_matrix,
        X_sample,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=15,
    )
    plt.title(
        f"{model_name} - Global Feature Importance", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values_matrix,
        X_sample,
        feature_names=feature_names,
        show=False,
        max_display=15,
    )
    plt.title(f"{model_name} - SHAP Value Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

    return shap_values, feature_names, top_features


def plot_shap_top20_comparison(
    shap_lgbm_top20: pd.Series,
    shap_xgb_top20: pd.Series,
    title: str = "Top 20 Feature Importance Comparison: LGBM vs XGB",
) -> None:
    """
    Plot a side-by-side horizontal bar chart comparing the top 20 SHAP feature
    importances for LGBM and XGB models.

    Parameters
    ----------
    shap_lgbm_top20 : pd.Series
        Series of mean absolute SHAP values (top 20) for LGBM, indexed by feature names.
    shap_xgb_top20 : pd.Series
        Series of mean absolute SHAP values (top 20) for XGB, indexed by feature names.
    title : str, optional
        Title for the plot (default is LGBM vs XGB comparison).

    Returns
    -------
    None
        Displays a matplotlib bar plot.
    """
    df_compare = pd.DataFrame(
        {"LGBM_SHAP": shap_lgbm_top20, "XGB_SHAP": shap_xgb_top20}
    ).fillna(0)

    df_compare = df_compare.loc[
        df_compare.sum(axis=1).sort_values(ascending=False).head(20).index
    ]

    df_compare = df_compare.sort_values("LGBM_SHAP", ascending=True)

    plt.figure(figsize=(12, 8))
    bar_width = 0.4
    indices = np.arange(len(df_compare))

    plt.barh(
        indices - bar_width / 2,
        df_compare["LGBM_SHAP"],
        height=bar_width,
        label="LGBM",
        color="#1f77b4",
    )
    plt.barh(
        indices + bar_width / 2,
        df_compare["XGB_SHAP"],
        height=bar_width,
        label="XGB",
        color="#ff7f0e",
    )

    plt.yticks(indices, df_compare.index)
    plt.xlabel("Mean |SHAP value|")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_threshold_optimization(y_true, y_probs, step=0.01):
    thresholds = np.arange(0.0, 1.01, step)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred))
        f1s.append(f1_score(y_true, y_pred))

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precisions, label="Precision", color="blue")
    plt.plot(thresholds, recalls, label="Recall", color="green")
    plt.plot(thresholds, f1s, label="F1 Score", color="red")
    plt.axvline(0.5, linestyle="--", color="gray", label="Default = 0.5")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Optimization for Ensemble")
    plt.legend()
    plt.grid(True)
    plt.show()

    best_f1_idx = np.argmax(f1s)
    print(f"Best F1: {f1s[best_f1_idx]:.3f} at threshold={thresholds[best_f1_idx]:.2f}")
    best_recall_idx = np.argmax(recalls)
    print(
        f"Max Recall: {recalls[best_recall_idx]:.3f} at threshold={thresholds[best_recall_idx]:.2f}"
    )
    best_precision_idx = np.argmax(precisions)
    print(
        f"Max Precision: {precisions[best_precision_idx]:.3f} at threshold={thresholds[best_precision_idx]:.2f}"
    )
