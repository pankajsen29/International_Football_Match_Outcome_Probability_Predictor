####################################################################
# Step 3: 
# 
# This module defines all machine learning models used for football 
# match outcome prediction. All models are returned in a dictionary so 
# that the training pipeline can iterate over them and compare their performance.
####################################################################
import src.config as cfg

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


###########################################################
# Returns a dictionary (model_name : sklearn_estimator) containing 
# all machine learning models used in this project 
########################################################
def get_models():
    models = {
        # Logistic Regression
        "Logistic Regression":
            LogisticRegression(
                random_state=cfg.RANDOM_STATE,                
                solver="lbfgs",
                max_iter=3000
            ),


        # Decision Tree
        "Decision Tree":
            DecisionTreeClassifier(
                random_state=cfg.RANDOM_STATE,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
            ),


        # Random Forest
        "Random Forest":
            RandomForestClassifier(
                random_state=cfg.RANDOM_STATE,
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                n_jobs=-1,
            ),


        # Gradient Boosting
        "Gradient Boosting":
            GradientBoostingClassifier(
                random_state=cfg.RANDOM_STATE,
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
            ),


        # XGBoost
        "XGBoost":
            XGBClassifier(
                random_state=cfg.RANDOM_STATE,
                objective="multi:softprob",
                num_class=3,
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="mlogloss",
                tree_method="hist",
            ),
    }
    return models


################## Display all available machine learning models ################
def print_available_models():
    print("\nAvailable Models:\n")

    for model_name in get_models():
        print(f" - {model_name}")