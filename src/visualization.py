####################################################################
# Step 6: Visualization of the evaluation results
#       - Model comparison chart
#       - Confusion matrix
####################################################################

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


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


def visualization_pipeline(results_df, detailed_results, y_test):
    print("\nGenerating visualizations...")

    # best model
    best_model = results_df.iloc[0]["Model"]
    evaluation = detailed_results[best_model]

    # compare all models
    plot_model_comparison(results_df)

    # best model visualizations
    plot_confusion_matrix(best_model, y_test, evaluation["Predictions"])    