
from src.preprocess import preprocess_dataset
from src.feature_engineering import feature_engineering_pipeline
from src.model import print_available_models
from src.train import train_pipeline
from src.evaluate import evaluate_pipeline
from src.visualization import visualization_pipeline
from src.inference.predict import prediction_pipeline
import src.config as cfg

# Step 1: cleans and standardizes the data
#preprocess_dataset()

# Step 2: creates the features
#feature_engineering_pipeline()

# Step 3: defines the ML models
#print_available_models()

# Step 4: execute train pipeline
#train_pipeline()

# Step 5: execute evaluate pipeline
#results_df, detailed_results, X_test, y_test =  evaluate_pipeline()

# Step 6: visualization of the evaluation results
#visualization_pipeline(results_df, detailed_results, y_test)

# Step 7: make predictions for a future match
prediction_pipeline(
    home_team="Brazil",
    away_team="Germany",
    tournament="FIFA World Cup",
    neutral=False,
    best_model=cfg.BEST_MODEL
)
