####################################################################
# Step 1: preprocesses data -> cleans and standardizes dataset.

# Input:
#   data/raw/results.csv
#   data/raw/former_names.csv

# Output:
#   data/cleaned/results_cleaned.csv
#   data/cleaned/upcoming_matches.csv
####################################################################

from pathlib import Path
import pandas as pd
import numpy as np


############# Paths ###################
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEAN_DIR = PROJECT_ROOT / "data" / "cleaned"

RESULTS_FILE = RAW_DIR / "results.csv"
FORMER_NAMES_FILE = RAW_DIR / "former_names.csv"

OUTPUT_FILE = CLEAN_DIR / "results_cleaned.csv"

UPCOMING_MATCHES_FILE = CLEAN_DIR / "upcoming_matches.csv"

CLEAN_DIR.mkdir(parents=True, exist_ok=True)

##############  Safe historical name changes ############# 
# included genuine country renamings, geopolitical splits are avoided, e.g.,:
# Yugoslavia -> Serbia
# Soviet Union -> Russia

SAFE_NAME_CHANGES = {
    "West Germany": "Germany",
    "Burma": "Myanmar",
    "Ceylon": "Sri Lanka",
    "Swaziland": "Eswatini"
}

############ Loads datasets ##################
def load_datasets():
    print("Loading datasets...")

    results = pd.read_csv(RESULTS_FILE)
    former_names = pd.read_csv(FORMER_NAMES_FILE)

    print(f"Matches loaded : {len(results):,}")
    print(f"Former names   : {len(former_names):,}")

    return results, former_names


############  Inspects dataset: shows dataset overview ############
def show_dataset_overview(dataframe):
    print("\n=========== DATASET OVERVIEW =============")

    print(dataframe.info())

    print("\nMissing Values")
    print(dataframe.isnull().sum()) # hint: pandas treats NA, NaN, N/A, NULL, null, #N/A etc. as missing values

    print("\nDuplicate Rows")
    print(dataframe.duplicated().sum())


############  Convert columns to their appropriate data types #############
def convert_data_types(dataframe):
    print("\nConverting columns to their appropriate data types...")

    # Date
    dataframe["date"] = pd.to_datetime(dataframe["date"])

    # Scores
    dataframe["home_score"] = pd.to_numeric(
        dataframe["home_score"],
        errors="coerce"
    )

    dataframe["away_score"] = pd.to_numeric(
        dataframe["away_score"],
        errors="coerce"
    )

    # Neutral venue
    dataframe["neutral"] = (
        dataframe["neutral"]
        .astype(str)
        .str.upper()
        .map({"TRUE": True, "FALSE": False})
    )
    return dataframe


############  Standardize Text Columns: removes leading/trailing whitespaces ############  
def clean_text_columns(dataframe):
    print("\nCleaning text columns...")
    text_columns = [
        "home_team",
        "away_team",
        "tournament",
        "city",
        "country"
    ]

    for column in text_columns:
        dataframe[column] = dataframe[column].astype(str).str.strip()

    return dataframe


############  Standardizes former country names  ############  
def standardize_team_names(dataframe, former_names):
    print("\nAvailable historical name mappings:")
    print(former_names)

    before_home = dataframe["home_team"].copy()
    before_away = dataframe["away_team"].copy()

    print("\nBut only below mappings are considered for replacements...")
    print(SAFE_NAME_CHANGES)

    dataframe["home_team"] = dataframe["home_team"].replace(SAFE_NAME_CHANGES)
    dataframe["away_team"] = dataframe["away_team"].replace(SAFE_NAME_CHANGES)

    home_changes = (before_home != dataframe["home_team"]).sum()
    away_changes = (before_away != dataframe["away_team"]).sum()

    print(f"\nHome team names updated : {home_changes}")
    print(f"Away team names updated : {away_changes}")

    return dataframe


############  Removes duplicate rows ############
def remove_duplicates(dataframe):
    before = len(dataframe)
    dataframe = dataframe.drop_duplicates()
    removed = before - len(dataframe)
    print(f"\nDuplicate rows removed: {removed}")
    return dataframe


############  Sorts chronologically ############
def sort_by_date(dataframe):
    dataframe = dataframe.sort_values("date").reset_index(drop=True)
    return dataframe


############  Handles missing values in the dataset ############
def handle_missing_values(dataframe):
    print("\nHandling missing values...")

    print("\nMissing Values Before Cleaning:")
    print(dataframe.isnull().sum())

    # separates upcoming matches
    upcoming_matches = dataframe[
        dataframe["home_score"].isna() &
        dataframe["away_score"].isna()
    ].copy()

    # need to remove rows with missing essential information
    essential_columns = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score"
    ]

    before = len(dataframe)

    dataframe = dataframe.dropna(subset=essential_columns)

    removed = before - len(dataframe)

    print(f"Removed {removed} rows with missing essential values.")

    # optionally filling text columns
    dataframe["tournament"] = dataframe["tournament"].fillna("Unknown")
    dataframe["city"] = dataframe["city"].fillna("Unknown")
    dataframe["country"] = dataframe["country"].fillna("Unknown")

    # optional
    dataframe["neutral"] = dataframe["neutral"].fillna("False")

    print("\nMissing Values After Cleaning:")
    print(dataframe.isnull().sum())

    return dataframe, upcoming_matches


############  Creates target variable ############
def create_target(dataframe):
    conditions = [
        dataframe["home_score"] > dataframe["away_score"],
        dataframe["home_score"] < dataframe["away_score"]
    ]

    choices = [
        "Home Win",
        "Away Win"
    ]

    dataframe["target"] = np.select(conditions, choices, default="Draw")

    return dataframe


############  Shows cleaning report ############
def print_summary(dataframe):
    print("\n=========== CLEANING SUMMARY =============")

    print(f"Final shape : {dataframe.shape}")

    print("\nTarget Distribution")
    print(dataframe["target"].value_counts())

    print("\nDate Range")
    print(dataframe["date"].min(), " -> ", dataframe["date"].max())


#############  Saves cleaned csv: results_cleaned.csv ############
def save_dataset(dataframe, file):
    dataframe.to_csv(file, index=False)
    print(f"\nCleaned dataset saved to:\n{file}")


#############  Calls all the cleaning steps in sequence ############
def preprocess_dataset():
    results, former_names = load_datasets()    

    results = convert_data_types(results)

    show_dataset_overview(results)

    results, upcoming_matches = handle_missing_values(results)

    results = clean_text_columns(results)

    results = standardize_team_names(results, former_names)

    results = remove_duplicates(results)

    results = sort_by_date(results)   

    results = create_target(results)

    print_summary(results)

    save_dataset(results, OUTPUT_FILE)
    save_dataset(upcoming_matches, UPCOMING_MATCHES_FILE)

