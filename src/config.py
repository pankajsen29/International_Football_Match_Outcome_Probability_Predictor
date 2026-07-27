####################################################################
# Step 0: defines the configurations.
####################################################################

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEAN_DIR = PROJECT_ROOT / "data" / "cleaned"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"

# input files
RESULTS_FILE = RAW_DIR / "results.csv"
FORMER_NAMES_FILE = RAW_DIR / "former_names.csv"

# output files
CLEANED_FILE = CLEAN_DIR / "results_cleaned.csv"
UPCOMING_MATCHES_FILE = CLEAN_DIR / "upcoming_matches.csv"
FEATURES_FILE = FEATURES_DIR / "matches_features.csv"

CLEAN_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "target"

# setting for ML algorithm
RANDOM_STATE = 42

# used in training loop
TEST_SIZE = 0.20
MODEL_DIR = PROJECT_ROOT / "checkpoints"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Model-specific Preprocessing Configuration
SCALE_NUMERICAL_FEATURES = {
    "Logistic Regression": True,
    "Decision Tree": False, 
    "Random Forest": False,
    "Gradient Boosting": False,
    "XGBoost": False
}

# Model display name mapping
MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "xgboost": "XGBoost"
}

# visualization settings
EVALUATION_RESULTS_DIR = PROJECT_ROOT / "results"
SAVE_FIGURES = True
SHOW_FIGURES = False # enable only during debugging