
####################################################################
# Step 4: Training pipeline for Football Match Outcome Predictor.

# Part 1:
#     - Load feature dataset,
#     - Separates input features and target,
#     - Split training/testing datasets.
# Part 2:
#     - For each ML model:
#           - Build preprocessing pipeline,
#           - Train it,
#           - Save that trained pipeline.
####################################################################

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

import src.config as cfg
from src.model import get_models

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
    dataframe["target"] = dataframe["target"].map(target_mapping)

    # separates the target
    y = dataframe["target"]

    # Drop columns NOT used (target: it is separated and should not be seen during training, date: is not relevant) for training
    columns_to_drop = ["target", "date"]
    X = dataframe.drop(columns=columns_to_drop)

    print(f"Number of features : {X.shape[1]}")
    print(f"Number of samples  : {X.shape[0]}")
    return X, y



# Splits dataset into training and testing datasets.
# We DO NOT shuffle.
# Football prediction is a time-series style problem.
# Older matches are used for training.
# Newer matches are used for testing.
#
# Returns:
#   X_train
#   X_test
#   y_train
#   y_test
def split_dataset(X, y):
    print("\nSplitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg.TEST_SIZE, shuffle=False)

    print(f"Training Matches : {len(X_train)}")
    print(f"Testing Matches  : {len(X_test)}")
    return X_train, X_test, y_train, y_test


def get_feature_category_details(X_train):
    print("\nFeature category details for building specific preprocessing steps...")
    # these columns contain categorries instead of numeric values 
    categorical_columns = ["home_team", "away_team", "tournament"]

    print("\nCategorical Features:")

    for column in categorical_columns:
        print(f"  {column}")

    # the remaining columns containing numeric values 
    numerical_columns = [column for column in X_train.columns
        if column not in categorical_columns]    

    print("\nNumerical Features:")
    print(len(numerical_columns))

    return categorical_columns, numerical_columns


# Build Preprocessing Pipeline
# Categorical columns
#    -> OneHotEncoder
# Numerical columns
#    -> StandardScaler for Logistic Regression, otherwise passed through unchanged
def build_preprocessor(categorical_columns, numerical_columns, scale_numeric_features = False):
    print("- Building preprocessing pipeline...")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns
            ),
            (
                "numerical",
                StandardScaler() if scale_numeric_features else "passthrough",
                numerical_columns
            )
        ]
    )
    return preprocessor


# Creates Training Pipeline by combineing the preprocessing and model into a single sklearn pipeline.
def create_training_pipeline(preprocessor, model):
    print("- Building training pipeline...")
    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                model
            )
        ]
    )
    return pipeline


# For each machine learning model: build the preprocessor and train it.
# Returns the dictionary containing trained sklearn Pipelines.
def train_models(X_train, y_train):
    print("\n========== Training Models ============")

    models = get_models()
    trained_models = {}
    counter = 1

    categorical_columns, numerical_columns = get_feature_category_details(X_train)

    for model_name, model in models.items():
        print(f"\n{counter}: Training {model_name}...")
        scale = cfg.SCALE_NUMERICAL_FEATURES.get(model_name, False)
        preprocessor = build_preprocessor(categorical_columns, numerical_columns, scale)
        train_pipeline = create_training_pipeline(preprocessor, model)
        train_pipeline.fit(X_train, y_train)
        trained_models[model_name] = train_pipeline
        counter = counter + 1
        print(f"- Training completed for {model_name}.")

    return trained_models

# Save every trained pipeline.
def save_models(trained_models):
    print("\nSaving Models...")

    for model_name, pipeline in trained_models.items():
        filename = model_name.lower().replace(" ", "_") + ".pkl"
        save_path = cfg.MODEL_DIR / filename
        joblib.dump(pipeline, save_path)
        print(f"{model_name} Saved at {save_path}")

# Complete Training Pipeline
def train_pipeline():

    # part 1: 
    dataframe = load_features(cfg.FEATURES_FILE)
    X, y = prepare_dataset(dataframe)
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # part 2:
    trained_models = train_models(X_train, y_train)
    save_models(trained_models)

    print("\nTraining is finished successfully...")

    return (trained_models, X_train, X_test, y_train, y_test)


