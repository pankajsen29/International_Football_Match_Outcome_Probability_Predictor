# International_Football_Match_Outcome_Probability_Predictor

A classical machine learning project that predicts the probability of possible match outcome (Home Win, Draw, Away Win) using historical football data of international matches. 

It focuses on structured tabular data and demonstrates the complete end-to-end machine learning workflow using Python and scikit-learn. The primary goal is to learn and apply classical machine learning concepts such as data preprocessing, feature engineering, model selection, hyperparameter tuning, and model evaluation.

## 1. Objectives
  - Build a football match outcome prediction model using historical match data.
  - Compare multiple classical machine learning algorithms to identify the best-performing model.
  - Understand how feature engineering influences predictive performance.
  - Evaluate models using appropriate classification metrics.
  - Create a reusable prediction pipeline that can later be deployed as a web application or API.

## 2. Choosing the dataset:
Football world cup is ongoing, so my idea is to use this for predicting the rest of the world cup matches, and also to use it for any future international matches. Hence I have chosen the below dataset:

"International Football Results.." from "https://www.kaggle.com/datasets"
[https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017]

because:
  - at the moment I would like to use this predictor for predicting world cup matches (meaning played between national teams),
  - and also to make it useable for predicting any other international matches,
  - famous and much larger training dataset which contains results over 40,000 international matches (including world cup matches, qualifiers, continental championships etc.),
  - national teams change less frequently than clubs, making feature engineering more manageable.

And not: 

football-data.co.uk: [https://www.football-data.co.uk/]

because:
  - this dataset is primarily for club competitions, not national teams; hence I won't be able to use it for predicting the ongoing world cup matches.

Also not: 

StatsBomb Open Data: [https://github.com/statsbomb/open-data]

because:
  - it is far too detailed for my classical ML project. It is mainly used by researchers and analysts.


## 3. Dataset details and usage:
[https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017/data]

This dataset includes Total 49,393 results of international football matches.

### DATASETS USED:

    results.csv includes the following columns:
    
        date - date of the match
        home_team - the name of the home team
        away_team - the name of the away team
        home_score - full-time home team score including extra time, not including penalty-shootouts
        away_score - full-time away team score including extra time, not including penalty-shootouts
        tournament - the name of the tournament
        city - the name of the city/town/administrative unit where the match was played
        country - the name of the country where the match was played
        neutral - TRUE/FALSE column indicating whether the match was played at a neutral venue
    
    former_names.csv includes the following columns:
    
        current - name of the team as is used currently (or the last name if the team does not exist anymore)
        former - former name used by said team
        start_date - start date of when former name was used
        end_date - end date of when former name was used

### DATASETS NOT USED:

    goalscorers.csv includes the following columns:
    
        date - date of the match
        home_team - the name of the home team
        away_team - the name of the away team
        team - name of the team scoring the goal
        scorer - name of the player scoring the goal
        own_goal - whether the goal was an own-goal
        penalty - whether the goal was a penalty
    
    Hint: goalscorers.csv dataset is not used as it is at the goal level, not at the match level. 
    But when predicting a future match, I would often not know whether a particular player will play or start, 
    whether he is injured, or whether he will score. Therefore, at the moment I am not using this dataset for my winner predictor.
      
    shootouts.csv includes the following columns:
    
        date - date of the match
        home_team - the name of the home team
        away_team - the name of the away team
        winner - winner of the penalty-shootout
        first_shooter - the team that went first in the shootout
    
     Hint: shootouts.csv dataset is not used as it contains only matches decided by penalties. 
     Also because my goal for my predictor is to predict at the end of regular time (or 90 minutes plus stoppage time) 
     and penalty shootouts occur only in knockout matches after a draw which are relatively rare. 
     Later I may use this if I plan to build a separate model to predict shootout winner.


## 3. Dataset Cleaning:

This step is for cleaning and standardizing the raw data.

**File:** dataset_preprocessing.py

**Outputs:**
    results_cleaned.csv
    upcoming_matches.csv

**Steps:**

A) Loads the input datasets: results.csv and former_names.csv.

B) Converts below columns to their appropriate data types:

    "dates" to "datetime"
    "scores" to "numeric"
    "neutral" to "Python booleans" by replacing "TRUE"/"FALSE"

C) Inspects/shows dataset overview:
 - shows the number of missing values along with column names. (hint: pandas treats NA, NaN, N/A, NULL, null, #N/A etc. as missing values)
 - also shows the number of duplicate rows.

D) Handles missing values:
 - previous step shows the number of missing values along with column names.
 - keeps the future matches in "upcoming_matches" dataframe.
 - removes rows with missing essential information:
   
       "date",
       "home_team",
       "away_team",
       "home_score",
       "away_score"
   
 - (optional) tournament, city, country are filled with "Unknown".
 - (optional) neutral is filled with "False".
 - returns the updated "results" dataframe.
 - also returns the "upcoming_matches" dataframe (which can be used for predictions later)
   
E) Standardizes text columns: removes leading/trailing spaces.

F) Standardize historical country names:

At the beginning it just shows the mapping available in "former_names.csv".
Columns like "start_date", "end_date" in this dataset are there to see the historical mappings. 
But I have only considered the safe historical name changes, meaning included only the genuine country renamings. 
Geopolitical splits (i.e., when dataset contains one country in two/three different names)
such as Yugoslavia -> Serbia, Soviet Union -> Russia etc. are not considered for replacements.

G) Removes duplicate rows.

H) Sorts matches chronologically (oldest -> newest)

I) Create the target variable 

    if home_score > away_score = "Home Win", 
    if home_score < away_score = "Away Win",
    Otherwise, "Draw"

J) prints the final cleaning report.

K) Saves results_cleaned.csv and upcoming_matches.csv


**Display Output:**

      Loading datasets...
      Matches loaded : 49,502
      Former names   : 36
      
      Converting columns to their appropriate data types...
      
      =========== DATASET OVERVIEW =============
      <class 'pandas.DataFrame'>
      RangeIndex: 49502 entries, 0 to 49501
      Data columns (total 9 columns):
       #   Column      Non-Null Count  Dtype         
      ---  ------      --------------  -----         
       0   date        49502 non-null  datetime64[us]
       1   home_team   49502 non-null  str           
       2   away_team   49502 non-null  str           
       3   home_score  49495 non-null  float64       
       4   away_score  49495 non-null  float64       
       5   tournament  49502 non-null  str           
       6   city        49502 non-null  str           
       7   country     49502 non-null  str           
       8   neutral     49502 non-null  bool          
      dtypes: bool(1), datetime64[us](1), float64(2), str(5)
      memory usage: 3.1 MB
      None
      
      Missing Values
      date          0
      home_team     0
      away_team     0
      home_score    7
      away_score    7
      tournament    0
      city          0
      country       0
      neutral       0
      dtype: int64
      
      Duplicate Rows
      0
      
      Handling missing values...
      
      Missing Values Before Cleaning:
      date          0
      home_team     0
      away_team     0
      home_score    7
      away_score    7
      tournament    0
      city          0
      country       0
      neutral       0
      dtype: int64
      Removed 7 rows with missing essential values.
      
      Missing Values After Cleaning:
      date          0
      home_team     0
      away_team     0
      home_score    0
      away_score    0
      tournament    0
      city          0
      country       0
      neutral       0
      dtype: int64
      
      Cleaning text columns...
      
      Available historical name mappings (only first 5 rows are shown):
                current                former  start_date    end_date
      0           Benin               Dahomey  1959-11-08  1975-11-30
      1    Burkina Faso           Upper Volta  1960-04-14  1984-08-04
      2         Curaçao  Netherlands Antilles  1957-03-03  2010-10-10
      3  Czechoslovakia               Bohemia  1903-04-05  1919-01-01
      4  Czechoslovakia   Bohemia and Moravia  1939-01-01  1945-05-01
      
      But only below mappings are considered for replacements...
      {'West Germany': 'Germany', 'Burma': 'Myanmar', 'Ceylon': 'Sri Lanka', 'Swaziland': 'Eswatini'}
      
      Home team names updated : 0
      Away team names updated : 0
      
      Duplicate rows removed: 0
      
      =========== CLEANING SUMMARY =============
      Final shape : (49495, 10)
      
      Target Distribution
      target
      Home Win    24256
      Away Win    13983
      Draw        11256
      Name: count, dtype: int64
      
      Date Range
      1872-11-30 00:00:00  ->  2026-07-04 00:00:00
      
      Cleaned dataset saved to:
      C:\xxxx\data\cleaned\results_cleaned.csv
      
      Cleaned dataset saved to:
      C:\xxxx\data\cleaned\upcoming_matches.csv
