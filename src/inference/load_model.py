####################################################################
# loads a trained pipeline
####################################################################

import joblib
import src.config as cfg

# Load a trained model
def load_model(best_model_name: str):
    model_path = cfg.MODEL_DIR / best_model_name
    print(f"\nLoading model: {model_path}")
    model = joblib.load(model_path)
    print("Model loaded successfully.")
    return model