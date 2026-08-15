####################################################################
# Step 7: Prediction Pipeline
#   - call create_prediction_features()
#   - receive feature DataFrame for prediction
#   - load saved model pipeline
#   - pipeline.predict()
#   - pipeline.predict_proba()
#   - prediction + probabilities
#
# Input:
#   Home team
#   Away team
#   Tournament
#   Neutral venue
#
# Output:
#   Predicted match result
#   Prediction probabilities
####################################################################

import src.config as cfg
from src.feature_engineering import create_prediction_features
from src.inference.load_model import load_model


# Make prediction
def predict_match(
    home_team,
    away_team,
    tournament,
    neutral,
    best_model="xgboost.pkl"
):
    print("\n=========== Football Match Prediction ===========")

    print(f"\nHome Team  : {home_team}")
    print(f"Away Team  : {away_team}")
    print(f"Tournament : {tournament}")
    print(f"Neutral    : {neutral}")

    # Step 1: Create features for the future match
    X_prediction = create_prediction_features(
        home_team=home_team,
        away_team=away_team,
        tournament=tournament,
        neutral=neutral
    )


    # Step 2: Load trained model pipeline
    #
    # The saved pipeline already contains:
    #   - preprocessing
    #   - classifier
    #
    # Therefore, we do NOT preprocess X_prediction manually.
    model = load_model(best_model)

    # Step 3: Predict class
    prediction = model.predict(X_prediction)

    # Step 4: Predict probabilities
    probabilities = model.predict_proba(X_prediction)

    # Extract single prediction
    predicted_class = prediction[0]

    # Extract probabilities for the single match
    predicted_probabilities = probabilities[0]

    # Step 5: Convert numeric class to meaningful label
    class_names = {0: "Home Win", 1: "Draw", 2: "Away Win"}

    predicted_result = class_names[predicted_class]

    # Step 6: Display prediction
    print("\nPrediction")
    print("-------------------------------------------------")
    print(f"\nPredicted Result : {predicted_result}")

    print("\nPrediction Probabilities:")

    for class_id, probability in zip(
        model.classes_,
        predicted_probabilities
    ):

        print(
            f"  {class_names[class_id]:10s} : "
            f"{probability:.2%}"
        )

    # Step 7: Return results
    return {
        "home_team": home_team,
        "away_team": away_team,
        "tournament": tournament,
        "neutral": neutral,
        "model": cfg.MODEL_DISPLAY_NAMES.get(best_model.replace(".pkl", ""), best_model),
        "prediction": predicted_result,
        "probabilities": {
            class_names[class_id]: probability
            for class_id, probability in zip(
                model.classes_,
                predicted_probabilities
            )
        }
    }

# Prediction Pipeline
def prediction_pipeline(home_team, away_team, tournament, neutral, best_model="xgboost.pkl"):

    print("\nStarting prediction pipeline...")

    result = predict_match(
        home_team=home_team,
        away_team=away_team,
        tournament=tournament,
        neutral=neutral,
        best_model=best_model
    )
    print("\nPrediction completed successfully.")
    return result