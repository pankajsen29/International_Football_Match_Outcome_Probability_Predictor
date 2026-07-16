####################################################################
# Step 2: creates the features

# Input:
#   data/raw/results_cleaned.csv

# Output:
#   data/features/matches_features.csv
####################################################################

from collections import defaultdict, deque
import pandas as pd
import src.config as cfg
import os

############ Create an empty statistics record for every team ###########
def initialize_team_stats():
    return defaultdict(
        lambda: {
            # Overall statistics
            "matches": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "goals_scored": 0,
            "goals_conceded": 0,

            # Rolling statistics (last 5 matches)
            "recent_results": deque(maxlen=5),
            "recent_goals_scored": deque(maxlen=5),
            "recent_goals_conceded": deque(maxlen=5),
        }
    )

######### Stores historical statistics for every pair of teams #######
def initialize_head_to_head():
    return defaultdict(
        lambda: {
            "matches": 0,
            "home_team_wins": 0,
            "away_team_wins": 0,
            "draws": 0,
            "home_team_goals": 0,
            "away_team_goals": 0,
        }
    )

#### Create a unique key regardless of home/away order #####
#### e.g., "France, Germany" / "Germany, France" >> ("France", "Germany")
def get_h2h_key(team1, team2):
    return tuple(sorted([team1, team2]))

####### Prevent divide-by-zero #########
def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0

    return numerator / denominator


############# Rolling Feature Calculation: Calculate statistics from the last five matches #######
def calculate_recent_statistics(stats):
    matches = len(stats["recent_results"])

    if matches == 0:
        return {
            "last5_win_rate": 0.0,
            "last5_avg_goals_scored": 0.0,
            "last5_avg_goals_conceded": 0.0,
            "last5_goal_difference": 0.0,
        }

    wins = stats["recent_results"].count("W")

    avg_scored = (sum(stats["recent_goals_scored"]) / matches)
    avg_conceded = (sum(stats["recent_goals_conceded"]) / matches)
    goal_difference = (sum(stats["recent_goals_scored"]) - sum(stats["recent_goals_conceded"])) / matches

    return {
        "last5_win_rate": wins / matches,
        "last5_avg_goals_scored": avg_scored,
        "last5_avg_goals_conceded": avg_conceded,
        "last5_goal_difference": goal_difference,
    }

############# Extract Features: Return historical statistics for one team ##########
def extract_team_features(team_name, team_stats):

    stats = team_stats[team_name]
    matches = stats["matches"]
    recent = calculate_recent_statistics(stats)

    return {

        # Career statistics
        "matches_played": matches,
        "win_rate": safe_divide(stats["wins"], matches),
        "draw_rate": safe_divide(stats["draws"], matches),
        "loss_rate": safe_divide(stats["losses"], matches),
        "avg_goals_scored": safe_divide(stats["goals_scored"], matches),
        "avg_goals_conceded": safe_divide(stats["goals_conceded"], matches),
        "goal_difference": safe_divide(
                stats["goals_scored"] -
                stats["goals_conceded"],
                matches
            ),

        # Last five matches
        **recent
    }

###### Return historical head-to-head statistics #######
def extract_head_to_head_features(home_team, away_team, h2h_stats):
    key = get_h2h_key(home_team, away_team)
    stats = h2h_stats[key]
    matches = stats["matches"]

    if matches == 0:
        return {
            "h2h_matches": 0,
            "h2h_home_team_win_rate": 0.0,
            "h2h_away_team_win_rate": 0.0,
            "h2h_draw_rate": 0.0,
            "h2h_home_avg_goals": 0.0,
            "h2h_away_avg_goals": 0.0,
            "h2h_goal_difference": 0.0,
        }

    # The stored statistics always refer to the
    # alphabetically first team and second team.
    # Convert them to the perspective of the current
    # home and away teams.

    first_team, second_team = key

    if home_team == first_team:

        home_wins = stats["home_team_wins"]
        away_wins = stats["away_team_wins"]

        home_goals = stats["home_team_goals"]
        away_goals = stats["away_team_goals"]

    else:

        home_wins = stats["away_team_wins"]
        away_wins = stats["home_team_wins"]

        home_goals = stats["away_team_goals"]
        away_goals = stats["home_team_goals"]

    return {
        "h2h_matches": matches,
        "h2h_home_team_win_rate": safe_divide(home_wins, matches),
        "h2h_away_team_win_rate": safe_divide(away_wins, matches),
        "h2h_draw_rate": safe_divide(stats["draws"], matches),
        "h2h_home_avg_goals": safe_divide(home_goals, matches),
        "h2h_away_avg_goals": safe_divide(away_goals, matches),
        "h2h_goal_difference": safe_divide(home_goals - away_goals, matches),
    }


######### Update team statistics after the features for the current match have been created #########
def update_team_statistics(home_team, away_team, home_score, away_score, team_stats):

    home = team_stats[home_team]
    away = team_stats[away_team]

    home["matches"] += 1
    away["matches"] += 1

    home["goals_scored"] += home_score
    home["goals_conceded"] += away_score

    away["goals_scored"] += away_score
    away["goals_conceded"] += home_score

    # Determine match result
    if home_score > away_score:

        home["wins"] += 1
        away["losses"] += 1

        home_result = "W"
        away_result = "L"

    elif home_score < away_score:

        away["wins"] += 1
        home["losses"] += 1

        home_result = "L"
        away_result = "W"

    else:

        home["draws"] += 1
        away["draws"] += 1

        home_result = "D"
        away_result = "D"

    # Update rolling history
    home["recent_results"].append(home_result)
    away["recent_results"].append(away_result)

    home["recent_goals_scored"].append(home_score)
    home["recent_goals_conceded"].append(away_score)

    away["recent_goals_scored"].append(away_score)
    away["recent_goals_conceded"].append(home_score)


######## Update H2H statistics after a match #########
def update_head_to_head(home_team, away_team, home_score, away_score, h2h_stats):
    key = get_h2h_key(home_team, away_team)
    stats = h2h_stats[key]
    first_team, second_team = key

    if home_team == first_team:
        stats["home_team_goals"] += home_score
        stats["away_team_goals"] += away_score

        if home_score > away_score:
            stats["home_team_wins"] += 1

        elif home_score < away_score:
            stats["away_team_wins"] += 1

        else:
            stats["draws"] += 1

    else:
        stats["home_team_goals"] += away_score
        stats["away_team_goals"] += home_score

        if away_score > home_score:
            stats["home_team_wins"] += 1

        elif away_score < home_score:
            stats["away_team_wins"] += 1

        else:
            stats["draws"] += 1

    stats["matches"] += 1


############### Create Feature Row: create one feature dictionary for a match ###########
def create_feature_row(match, team_stats, h2h_stats):
    home = extract_team_features(match.home_team, team_stats)
    away = extract_team_features(match.away_team, team_stats)    
    h2h = extract_head_to_head_features(match.home_team, match.away_team, h2h_stats)
    
    return {
        "date": match.date,
        "home_team": match.home_team,
        "away_team": match.away_team,
        "tournament": match.tournament,
        "neutral": match.neutral,

        # Career statistics
        "home_matches_played": home["matches_played"],
        "away_matches_played": away["matches_played"],
        "home_win_rate": home["win_rate"],
        "away_win_rate": away["win_rate"],
        "home_avg_goals": home["avg_goals_scored"],
        "away_avg_goals": away["avg_goals_scored"],
        "home_avg_conceded": home["avg_goals_conceded"],
        "away_avg_conceded": away["avg_goals_conceded"],
        "home_goal_difference": home["goal_difference"],
        "away_goal_difference": away["goal_difference"],

        # Last five matches
        "home_last5_win_rate": home["last5_win_rate"],
        "away_last5_win_rate": away["last5_win_rate"],
        "home_last5_avg_goals": home["last5_avg_goals_scored"],
        "away_last5_avg_goals": away["last5_avg_goals_scored"],
        "home_last5_avg_conceded": home["last5_avg_goals_conceded"],
        "away_last5_avg_conceded": away["last5_avg_goals_conceded"],
        "home_last5_goal_difference": home["last5_goal_difference"],
        "away_last5_goal_difference": away["last5_goal_difference"],

        # head to head statistics
        **h2h,

        "target": match.target
    }


############# Load Dataset ####################
def load_dataset(filepath):
    print("\nLoading Cleaned Dataset...")

    dataframe = pd.read_csv(filepath)
    dataframe["date"] = pd.to_datetime(dataframe["date"])
    dataframe = dataframe.sort_values("date").reset_index(drop=True)
    print(f"\nLoaded {len(dataframe)} matches.")

    return dataframe


########################################################
# generates machine learning features for every match. 
# Features are always calculated using only matches that occurred BEFORE the current match.
###########################################################
def generate_features(dataframe):
    print("\nGenerating Features...")
    team_stats = initialize_team_stats()
    h2h_stats = initialize_head_to_head()

    feature_rows = []
    total_matches = len(dataframe)

    for index, match in enumerate(dataframe.itertuples(index=False), start=1):

        # Create feature row
        feature_row = create_feature_row(match, team_stats, h2h_stats)

        feature_rows.append(feature_row)

        # Update statistics AFTER feature creation
        update_team_statistics(
            home_team=match.home_team,
            away_team=match.away_team,
            home_score=match.home_score,
            away_score=match.away_score,
            team_stats=team_stats
        )

        update_head_to_head(
            home_team=match.home_team,
            away_team=match.away_team,
            home_score=match.home_score,
            away_score=match.away_score,
            h2h_stats=h2h_stats
        )

        # Progress
        if index % 1000 == 0 or index == total_matches:
            print(f"Processed {index}/{total_matches} matches")

    feature_dataframe = pd.DataFrame(feature_rows)

    return feature_dataframe



########## Display Summary #######################
def print_summary(feature_dataframe):
    print("\n=========== FEATURE ENGINEERING SUMMARY ===========\n")

    print(f"Number of matches : {len(feature_dataframe)}")
    print(f"Number of features: {len(feature_dataframe.columns)}")
    print("\nFeature Columns:\n")

    for column in feature_dataframe.columns:
        print(f" - {column}")

    print("\nPreview:\n")
    print(feature_dataframe.head())



################ Save Dataset ##################
def save_dataset(feature_dataframe, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    feature_dataframe.to_csv(filepath, index=False)
    print(f"\nFeature dataset saved successfully at: {filepath}")


########### entire pipeline #############
def feature_engineering_pipeline():
    print("Feature Engineering is being started...")

    dataframe = load_dataset(cfg.CLEANED_FILE)

    feature_dataframe = generate_features(dataframe)

    print_summary(feature_dataframe)

    save_dataset(feature_dataframe, cfg.FEATURES_FILE)

    print("\nFeature engineering completed successfully.")