####################################################################
# Step 5: Loads trained models and evaluates them on the test dataset.

#   - Dataset preparation Pipelines
#       - Load Feature Dataset
#       - Prepare Dataset
#       - Split Dataset
#   - Load Trained Pipelines
#   - FOR each trained model
#       - Predict Test Set
#       - Calculate Metrics
#       - Print Results
#       - Select Best Model
#       - Plot Confusion Matrix
#       - Plot ROC Curves

# Hint: no preprocessing is done here, as every saved pipeline already 
# contains (preprocessor + classifier)
####################################################################

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss
)

import src.config as cfg


# Loads engineered feature dataset.
# returns: pandas.DataFrame
def load_features(csv_path):
    print("\nLoading feature dataset...")
    dataframe = pd.read_csv(csv_path)
    print(f"Loaded {len(dataframe)} matches")
    return dataframe


# Separates input features (X) and target labels (y)
# Returns:
#    X : pandas.DataFrame -> input features
#    y : pandas.Series -> correct answers
def prepare_dataset(dataframe):
    print("\nPreparing dataset...")
    
    # original dataframe should not change
    dataframe = dataframe.copy()

    # mapping of target labels to numbers so that ML can learn from
    target_mapping = {"Home Win": 0, "Draw": 1, "Away Win": 2}

    # Label encoding: replaces every target value with the mapping numbers
    dataframe[cfg.TARGET_COLUMN] = dataframe[cfg.TARGET_COLUMN].map(target_mapping)

    # verify that every label was recognized (example: suppose one row accidentally contains "Home win" instead of "Home Win", .map() above will produce NaN)
    if dataframe[cfg.TARGET_COLUMN].isnull().any():
        raise ValueError("Unknown target labels found.")

    # separates the target
    y = dataframe[cfg.TARGET_COLUMN]

    # Drop columns NOT used (target: it is separated and should not be seen during training, date: is not relevant) for training
    columns_to_drop = [cfg.TARGET_COLUMN, "date"]
    X = dataframe.drop(columns=columns_to_drop)

    print(f"Number of features : {X.shape[1]}")
    print(f"Number of samples  : {X.shape[0]}")
    return X, y


# Splits dataset into training and testing datasets.
# - We DO NOT shuffle.
# - Football prediction is a time-series style problem.
# - Older matches are used for training.
# - Newer matches are used for testing.
# Returns:
#   X_train
#   X_test
#   y_train
#   y_test
#
# [Hint: exact same function as used during training]
def split_dataset(X, y):
    print("\nSplitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.TEST_SIZE, shuffle=False)

    print(f"Training Matches : {len(X_train)}")
    print(f"Testing Matches  : {len(X_test)}")
    return X_test, y_test


# Complete dataset preparation pipeline by combining the above functions
def prepare_test_data():
    dataframe = load_features(cfg.FEATURES_FILE)
    X, y = prepare_dataset(dataframe)
    X_test, y_test = split_dataset(X, y)
    return X_test, y_test


# Load all trained model pipelines from the models directory.
# Returns: dict: Dictionary containing all trained pipelines.
def load_models():
    print("\nLoading trained models...")
    models = {}

    # Find all .pkl files inside the models directory
    model_files = sorted(cfg.MODEL_DIR.glob("*.pkl"))

    if not model_files:
        raise FileNotFoundError(f"No trained models found in: {cfg.MODEL_DIR}")

    for model_path in model_files:
        # Example: random_forest.pkl = Random Forest
        model_key = model_path.stem # "stem" returns filename without extension
        if model_key not in cfg.MODEL_DISPLAY_NAMES:
            raise KeyError(f"No display name configured for '{model_key}'.")

        model_name = cfg.MODEL_DISPLAY_NAMES.get(model_key)
        pipeline = joblib.load(model_path) # loads the entire trained pipeline
        models[model_name] = pipeline
        print(f"Loaded: {model_name}")

    print(f"\nTotal Models Loaded: {len(models)}")

    return models


# Evaluates a single trained model
# Args:
#         model_name (str): Name of the model.
#         pipeline: Trained sklearn pipeline.
#         X_test (DataFrame): Test features.
#         y_test (Series): True labels.
# returns:  dict: Evaluation metrics.
def evaluate_model(model_name, pipeline, X_test, y_test):
    print(f"\nEvaluating {model_name}:")

    # Predict class labels
    y_pred = pipeline.predict(X_test)

    # Predict class probabilities
    y_prob = pipeline.predict_proba(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    logloss = log_loss(y_test, y_prob)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"Log Loss : {logloss:.4f}")

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Log Loss": logloss,
        "Predictions": y_pred,
        "Probabilities": y_prob
    }


# Evaluate all trained models.
# Args:
#         models (dict): Dictionary of trained pipelines.
#         X_test (DataFrame): Test features.
#         y_test (Series): True labels.

#     Returns:
#         DataFrame: Evaluation summary.
#         dict: Complete evaluation results.
def evaluate_all_models(models, X_test, y_test):
    print("\nEvaluating All Models...")

    results = []
    detailed_results = {}

    for model_name, pipeline in models.items():

        evaluation = evaluate_model(
            model_name,
            pipeline,
            X_test,
            y_test
        )

        results.append({
            "Model": evaluation["Model"],
            "Accuracy": evaluation["Accuracy"],
            "Precision": evaluation["Precision"],
            "Recall": evaluation["Recall"],
            "F1 Score": evaluation["F1 Score"],
            "Log Loss": evaluation["Log Loss"]
        })

        detailed_results[model_name] = evaluation

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="Log Loss",
        ascending=True
    ).reset_index(drop=True)

    return results_df, detailed_results


# Print evaluation summary for all models
def print_summary(results_df):
    print("\nModel Comparison:")
    print(results_df.round(4))
    best_model = results_df.iloc[0]

    print("\nBest Model (Primary Ranking Metric = Log Loss)")
    print("-------------------------------------------------")
    print(f"Model    : {best_model['Model']}")
    print(f"Accuracy : {best_model['Accuracy']:.4f}")
    print(f"F1 Score : {best_model['F1 Score']:.4f}")
    print(f"Log Loss : {best_model['Log Loss']:.4f}")


# main evaluate pipeline
def evaluate_pipeline():
    print("\n======= Model Evaluation ============")

    # Step 1: Load and prepare feature engineered dataset
    X_test, y_test = prepare_test_data()

    # Step 2: Load trained models
    models = load_models()

    # Step 3: Evaluate all models
    results_df, detailed_results = evaluate_all_models(
        models,
        X_test,
        y_test
    )

    # Step 4: Print evaluation summary
    print_summary(results_df)

    print("\nEvaluation completed successfully.")


