####################################################################
# Step 0: defines the configurations.
####################################################################

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEAN_DIR = PROJECT_ROOT / "data" / "cleaned"

RESULTS_FILE = RAW_DIR / "results.csv"
FORMER_NAMES_FILE = RAW_DIR / "former_names.csv"

OUTPUT_FILE = CLEAN_DIR / "results_cleaned.csv"

UPCOMING_MATCHES_FILE = CLEAN_DIR / "upcoming_matches.csv"

CLEAN_DIR.mkdir(parents=True, exist_ok=True)