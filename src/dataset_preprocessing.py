####################################################################
# Step 1: preprocesses data -> cleans and standardizes dataset.

# Input:
#   data/raw/results.csv
#   data/raw/former_names.csv

# Output:
#   data/cleaned/results_cleaned.csv
####################################################################

from pathlib import Path
import pandas as pd


############# Paths ###################
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEAN_DIR = PROJECT_ROOT / "data" / "cleaned"

RESULTS_FILE = RAW_DIR / "results.csv"
FORMER_NAMES_FILE = RAW_DIR / "former_names.csv"

OUTPUT_FILE = CLEAN_DIR / "results_cleaned.csv"

CLEAN_DIR.mkdir(parents=True, exist_ok=True)



############ Loads datasets ##################
def load_data():
    print("Loading datasets...")

    results = pd.read_csv(RESULTS_FILE)
    former_names = pd.read_csv(FORMER_NAMES_FILE)

    print(f"Matches loaded : {len(results):,}")
    print(f"Former names   : {len(former_names):,}")

    return results, former_names


############  Inspects Dataset: shows dataset overview ############
def show_dataset_overview(dataframe):
    print("\n=========== DATASET OVERVIEW =============")

    print(dataframe.info())

    print("\nMissing Values")
    print(dataframe.isnull().sum())

    print("\nDuplicate Rows")
    print(dataframe.duplicated().sum())


def main():
    results, former_names = load_data()
    show_dataset_overview(results)

if __name__ == "__main__":
    main()