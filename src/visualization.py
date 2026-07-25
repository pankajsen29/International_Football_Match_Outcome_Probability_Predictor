####################################################################
# Step 6: Visualization of the evaluation results
#       - Model comparison chart
####################################################################


import matplotlib.pyplot as plt

# Plot comparison of all models.
def plot_model_comparison(results_df):
    plt.figure(figsize=(10,6))
    plt.bar(results_df["Model"], results_df["Accuracy"] )
    plt.title("Model Accuracy Comparison")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=20)
    plt.tight_layout()

    plt.show()