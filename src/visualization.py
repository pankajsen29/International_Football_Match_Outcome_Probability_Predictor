####################################################################
# Step 6: Visualization of the evaluation results
#       - Model comparison chart
#       - Confusion matrix
#       - ROC curve
#       - Calibration curve
#       - Feature importance
####################################################################

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
    plt.figure(figsize=(10,6))
    plt.bar(results_df["Model"], results_df["Accuracy"] )
    plt.title("Model Accuracy Comparison")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=20)
    plt.tight_layout()

    plt.show()

# plots the confusion matrix for standard classification evaluation
def plot_confusion_matrix(model_name, y_test, y_pred):

    plt.figure(figsize=(6,6))
    ConfusionMatrixDisplay.from_predictions( 
        y_test,
        y_pred,
        display_labels=["Home Win", "Draw", "Away Win"],
        cmap="Blues"

    )
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()
    plt.show()


# plots multiclass ROC curve (one-vs-rest) which shows probability discrimination.
def plot_roc_curve(model_name, y_test, y_prob):
    y_test_bin = label_binarize(
        y_test,
        classes=[0,1,2]
    )

    plt.figure(figsize=(7,6))
    class_names = ["Home Win", "Draw", "Away Win"]

    for i in range(3):
        fpr, tpr, _ = roc_curve(
            y_test_bin[:,i],
            y_prob[:,i]
        )

        roc_auc = auc(fpr,tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.3f})")

    plt.plot([0,1], [0,1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()


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

    indices = np.argsort(importance)[::-1][:top_n]

    plt.figure(figsize=(10,6))
    plt.barh(
        np.array(feature_names)[indices][::-1],
        importance[indices][::-1]
    )
    plt.title(f"{model_name} Feature Importance")
    plt.tight_layout()
    plt.show()


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

    plt.figure(figsize=(8, 6))

    for i in range(len(class_names)):

        CalibrationDisplay.from_predictions(
            y_true=y_test_bin[:, i],
            y_prob=y_prob[:, i],
            n_bins=10,
            strategy="uniform",
            name=class_names[i]
        )

    # Perfect calibration line
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.title(f"{model_name} Calibration Curve")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Frequency")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


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