
from src.preprocess import preprocess_dataset
from src.feature_engineering import feature_engineering_pipeline
from src.model import print_available_models
from src.train import train_pipeline
from src.evaluate import evaluate_pipeline
from src.visualization import visualization_pipeline

# Step 1: cleans and standardizes the data
#preprocess_dataset()

# Step 2: creates the features
#feature_engineering_pipeline()

# Step 3: defines the ML models
#print_available_models()

# Step 4: execute train pipeline
#train_pipeline()

# Step 5: execute evaluate pipeline
results_df, detailed_results, X_test, y_test =  evaluate_pipeline()

# Step 6: visualization of the evaluation results
visualization_pipeline(results_df, detailed_results, y_test)