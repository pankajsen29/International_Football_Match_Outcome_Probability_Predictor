####################################################################
# Step 6: Visualization of the evaluation results
#       - Model comparison chart
#       - Confusion matrix
#       - ROC curve
#       - Calibration curve
#       - Feature importance
####################################################################

import matplotlib
import src.config as cfg
if not cfg.SHOW_FIGURES:
    matplotlib.use("Agg") # force a non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)
from sklearn.calibration import CalibrationDisplay
from sklearn.preprocessing import label_binarize



# Plots model comparison bar chart for easy comparison of all the models.
def plot_model_comparison(results_df):
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(results_df["Model"], results_df["Accuracy"] )
    ax.set_title("Model Accuracy Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()

    if cfg.SAVE_FIGURES:
        fig.savefig(
            cfg.EVALUATION_RESULTS_DIR / "model_comparison.png",
            dpi=300,
            bbox_inches="tight"
        )
    if cfg.SHOW_FIGURES:
        plt.show()
    plt.close(fig)

# plots the confusion matrix for standard classification evaluation
def plot_confusion_matrix(model_name, y_test, y_pred):

    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions( 
        y_test,
        y_pred,
        display_labels=["Home Win", "Draw", "Away Win"],
        cmap="Blues",
        ax=ax
    )
    ax.set_title(f"{model_name} Confusion Matrix")
    fig.tight_layout()
    if cfg.SAVE_FIGURES:
        fig.savefig(
            cfg.EVALUATION_RESULTS_DIR / "confusion_matrix.png",
            dpi=300,
            bbox_inches="tight"
        )
    if cfg.SHOW_FIGURES:
        plt.show()
    plt.close(fig)


# plots multiclass ROC curve (one-vs-rest) which shows probability discrimination.
def plot_roc_curve(model_name, y_test, y_prob):
    y_test_bin = label_binarize(
        y_test,
        classes=[0,1,2]
    )

    fig, ax = plt.subplots(figsize=(7,6))
    class_names = ["Home Win", "Draw", "Away Win"]

    for i in range(3):
        fpr, tpr, _ = roc_curve(
            y_test_bin[:,i],
            y_prob[:,i]
        )

        roc_auc = auc(fpr,tpr)
        ax.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.3f})")

    ax.plot([0,1], [0,1], "--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} ROC Curve")
    ax.legend()
    fig.tight_layout()
    if cfg.SAVE_FIGURES:
        fig.savefig(
            cfg.EVALUATION_RESULTS_DIR / "roc_curve.png",
            dpi=300,
            bbox_inches="tight"
        )
    if cfg.SHOW_FIGURES:
        plt.show()
    plt.close(fig)


# Feature Importance (essential for tree-based models) indicates which 
# features influenced the prediction most. This is only applicable to:
# - Random Forest
# - Gradient Boosting
# - XGBoost
def plot_feature_importance(model_name, pipeline, top_n=20):

    classifier = pipeline.named_steps["classifier"]
    if not hasattr(classifier,"feature_importances_"):
        return

    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    importance = classifier.feature_importances_

    # limit top_n to the number of available features
    top_n = min(top_n, len(importance))

    # indices of the top_n most important features
    indices = np.argsort(importance)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10,6))
    ax.barh(
        np.array(feature_names)[indices][::-1],
        importance[indices][::-1]
    )
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Features")
    ax.set_title(f"{model_name} Feature Importance")
    fig.tight_layout()
    if cfg.SAVE_FIGURES:
        fig.savefig(
            cfg.EVALUATION_RESULTS_DIR / "feature_importance.png",
            dpi=300,
            bbox_inches="tight"
        )
    if cfg.SHOW_FIGURES:
        plt.show()
    plt.close(fig)


# Plot calibration curves for multiclass classification using 
# a One-vs-Rest approach. This tells whether the predicted probabilities 
# can be trusted.
# Args:
#         model_name (str): Name of the model.
#         y_test (Series): True labels.
#         y_prob (ndarray): Predicted probabilities.
def plot_calibration_curve(model_name, y_test, y_prob):
    print(f"\nGenerating calibration curve for {model_name}...")

    class_names = ["Home Win", "Draw", "Away Win"]

    # Convert multiclass labels to binary labels
    y_test_bin = label_binarize(
        y_test,
        classes=[0, 1, 2]
    )

    fig, ax = plt.subplots(figsize=(8,6))

    for i in range(len(class_names)):
        CalibrationDisplay.from_predictions(
            y_true=y_test_bin[:, i],
            y_prob=y_prob[:, i],
            n_bins=10,
            strategy="uniform",
            name=class_names[i],
            ax=ax
        )

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax.set_title(f"{model_name} Calibration Curve")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    if cfg.SAVE_FIGURES:
        fig.savefig(
            cfg.EVALUATION_RESULTS_DIR / "calibration_curve.png",
            dpi=300,
            bbox_inches="tight"
        )
    if cfg.SHOW_FIGURES:
        plt.show()
    plt.close(fig)



def visualization_pipeline(results_df, detailed_results, y_test):
    print("\nGenerating visualizations...")

    # best model
    best_model = results_df.iloc[0]["Model"]
    evaluation = detailed_results[best_model]

    # compare all models
    plot_model_comparison(results_df)

    # best model visualizations
    plot_confusion_matrix(best_model, y_test, evaluation["Predictions"])
    plot_roc_curve(best_model, y_test, evaluation["Probabilities"])
    plot_calibration_curve(best_model, y_test, evaluation["Probabilities"])
    plot_feature_importance(best_model, evaluation["Pipeline"])