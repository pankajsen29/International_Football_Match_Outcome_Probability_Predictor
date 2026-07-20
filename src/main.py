
from src.preprocess import preprocess_dataset
from src.feature_engineering import feature_engineering_pipeline
from src.model import print_available_models
from src.train import create_model_preprocessing_pipeline
# Step 1: cleans and standardizes the data
#preprocess_dataset()

# Step 2: creates the features
#feature_engineering_pipeline()

# Step 3: defines the ML models
print_available_models()

# Step 4: builds the model preprocessor pipeline
create_model_preprocessing_pipeline