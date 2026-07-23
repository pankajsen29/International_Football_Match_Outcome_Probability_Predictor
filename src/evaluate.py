####################################################################
# Step 5: Loads trained models and evaluates them on the test dataset.

#   - Dataset preparation Pipelines
#       - Load Feature Dataset
#       - Prepare Dataset
#       - Split Dataset
#   - Load Trained Pipelines
#   - FOR each trained model
#       - evaluate and calculate Metrics
#       - Print Results

# Hint: no preprocessing is done here, as every saved pipeline already 
# contains (preprocessor + classifier)
####################################################################

import pandas as pd
from sklearn.model_selection import train_test_split
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


# Complete dataset preparation pipeline
def prepare_test_data():
    dataframe = load_features(cfg.FEATURES_FILE)
    X, y = prepare_dataset(dataframe)
    X_test, y_test = split_dataset(X, y)
    return X_test, y_test


def evaluate():
    X_test, y_test = prepare_test_data()