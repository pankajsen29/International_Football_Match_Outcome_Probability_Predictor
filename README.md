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


## 4. Dataset Cleaning:

This step is for cleaning and standardizing the raw data.

**File:** preprocessing.py

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
      C:\xxxx\International_Football_Match_Outcome_Probability_Predictor\data\cleaned\results_cleaned.csv
      
      Cleaned dataset saved to:
      C:\xxxx\International_Football_Match_Outcome_Probability_Predictor\data\cleaned\upcoming_matches.csv


## 5. Feature Engineering pipeline:

This step creates the features the model can learn from.

**File:** feature_engineering.py

**Output:** matches_features.csv

Hint: The engineered features will have 3 categories:

1. Career statistics (e.g., overall win rate) capture overall team strength, long-term quality and stability.
   
Features computed:

    Matches played
    Career win rate
    Career draw/loss rate
    Average goals scored
    Average goals conceded
    Goal difference

2. Rolling recent-form features (e.g., last 5 match statistics) capture short-term trends, momentum, or a temporary slump.
   
Features computed:

    Last 5 win rate
    Last 5 average goals scored
    Last 5 average goals conceded
    Last 5 goal difference

3. Head-to-head statistics of team pair (meaning how the two teams have performed against each other in previous meetings) are among the strongest predictors in international football and would make the project significantly more realistic.
   
Features computed:

    H2H matches played
    H2H home team win rate
    H2H away team win rate
    H2H draw rate
    H2H average goals for each team
    H2H goal difference


**Steps (main pipeline):**

**A) load_dataset():** It reads cleaned dataset (i.e., results_cleaned.csv), converts date, sorts chronologically.

**B) generate_features():** It generates machine learning features for every match using only historical information (no future information).

    Step 1: initializations:
    
    	1.1: Initializes Team Statistics: initialize_team_stats()
    	  - creates default dictionary that stores cumulative and recent statistics for every team.
    
    	1.2: Initializes Head-To-Head (H2H) Statistics: initialize_head_to_head()
    	  - creates default dictionary that stores H2H statistics for every team pair.
    
    Step 2: Process Every Match (Chronological Order):
    
        - Step 2.1: Extract Features: create_feature_row(): creates one feature dictionary representing the current match.
        
            Internally, it calls:
      
            extract_team_features(home_team) - which calculates historical career statistics and then internally calls calculate_recent_statistics() to calculate the statistics for last 5 matches for home_team
      
            extract_team_features(away_team) - which calculates historical career statistics and then internally calls calculate_recent_statistics() to calculate the statistics for last 5 matches for away_team
      
            extract_head_to_head_features(home_team, away_team) - which calculates historical head-to-head statistics of this team-pair.
      
            That means for each team, it retrieves:
            
            Career statistics:
            
            Matches played
            Win rate
            Draw rate
            Loss rate
            Average goals scored
            Average goals conceded
            Goal difference
            
            Recent form (last 5 matches):
            
            Last 5 win rate
            Last 5 average goals scored
            Last 5 average goals conceded
            Last 5 goal difference
            
            H2H statistics: (of these two teams)
            
            H2H matches played
            H2H home team win rate
            H2H away team win rate
            H2H draw rate
            H2H average goals for each team
            H2H goal difference
            
            The important point here is, all of these statistics are calculated using only matches played before the current match, which prevents data leakage.
            
            The result is one dictionary like:
            
            {
                "home_team": "France",
                "away_team": "Germany",
                "home_win_rate": 0.62,
                "away_win_rate": 0.58,
                ...
                "target": "Home Win"
            }
            
            This dictionary will become one row of the final machine learning dataset.
    
        
        - Step 2.2: Append this feature dictionary to feature_rows.
        
        - Step 2.3: 
    
    	      A) Update team statistics: update_team_statistics()
    	          - After the features for the current match have been created and stored in feature_rows, the historical statistics for each team is updated.    
    
                Example:
                
                Current match:
                
                France 2–1 Germany
                
                Before the update:
                
                France:
                Matches = 10
                Wins = 6
                Goals = 18
                
                After the update:
                
                France:
                Matches = 11
                Wins = 7
                Goals = 20
                
                These updated statistics will be used when processing the next match involving France.
    
            B) Update head to head statistics: update_head_to_head()
                -  After processing a match, update the Head-to-Head (H2H) record between the two teams so it can be used as historical information for their future meetings. The update happens after feature extraction to prevent data leakage.
        
                Example
                
                Suppose these are the matches between France and Germany.
                
                Match 1 (2018)
                France 2 - 1 Germany
                
                Before this match:
                
                H2H Statistics
                
                Matches = 0
                France Wins = 0
                Germany Wins = 0
                Draws = 0
                France Goals = 0
                Germany Goals = 0
                
                Since this is the first meeting, the generated features are:
                
                h2h_matches = 0
                h2h_home_team_win_rate = 0
                h2h_away_team_win_rate = 0
                
                Only after creating the feature row do we update the H2H statistics.
                
                The stored statistics become:
                
                Matches = 1
                France Wins = 1
                Germany Wins = 0
                Draws = 0
                France Goals = 2
                Germany Goals = 1
                
                
                Match 2 (2021)
                Germany 0 - 0 France
                
                Now, when creating features for this match, the model already knows the previous meeting.
                
                Generated H2H features become:
                
                h2h_matches = 1
                France Win Rate = 1.0
                Germany Win Rate = 0.0
                Draw Rate = 0.0
                France Avg Goals = 2.0
                Germany Avg Goals = 1.0
                
                Only after feature extraction do we update the H2H statistics again.
                
                The updated H2H record becomes:
                
                Matches = 2
                France Wins = 1
                Germany Wins = 0
                Draws = 1
                France Goals = 2
                Germany Goals = 1
                
                
                Match 3 (2024)
                France vs Germany
                
                Now the generated H2H features are:
                
                h2h_matches = 2
                France Win Rate = 0.50
                Germany Win Rate = 0.00
                Draw Rate = 0.50
                France Avg Goals = 1.0
                Germany Avg Goals = 0.5
                
                These statistics summarize all previous meetings before the current match.
                
                
                Why is this important?
                
                Head-to-head statistics capture information that overall team statistics may miss.
                
                For example:
                
                Overall Win Rate
                
                France = 72%
                Germany = 68%
                
                These suggest the teams are similarly strong.
                
                However, their direct history may be:
                
                Previous Meetings
                
                France Wins = 7
                Germany Wins = 2
                Draws = 1
                
                This indicates that France has historically performed better specifically against Germany, even though both teams have strong overall records.
                
                By including H2H features, the model learns patterns specific to each rivalry, which can improve prediction accuracy compared with relying only on overall team performance.


    Step 3: Create the Feature DataFrame after every match has been processed. That means: Convert feature_rows -> DataFrame, with this step, the list of feature dictionaries is converted into a pandas DataFrame.
    
    Step 4: Return DataFrame


**C) print_summary():** displays the summary of feature engineering and the glimpse of engineered features.

**D) save_dataset():** Creates folders if necessary and saves DataFrame to matches_features.csv.
 
**Other functions:**

      - safe_divide(): this is to prevent divide-by-zero while calculating the statistics.
      - get_h2h_key(): It create a unique key regardless of home/away team order.
      
**Display Output:**

      Feature Engineering is being started...
      
      Loading Cleaned Dataset...
      
      Loaded 49495 matches.
      
      Generating Features...
      Processed 1000/49495 matches
      Processed 2000/49495 matches
      Processed 3000/49495 matches
      Processed 4000/49495 matches
      Processed 5000/49495 matches
      Processed 6000/49495 matches
      Processed 7000/49495 matches
      Processed 8000/49495 matches
      Processed 9000/49495 matches
      Processed 10000/49495 matches
      Processed 11000/49495 matches
      Processed 12000/49495 matches
      Processed 13000/49495 matches
      Processed 14000/49495 matches
      Processed 15000/49495 matches
      Processed 16000/49495 matches
      Processed 17000/49495 matches
      Processed 18000/49495 matches
      Processed 19000/49495 matches
      Processed 20000/49495 matches
      Processed 21000/49495 matches
      Processed 22000/49495 matches
      Processed 23000/49495 matches
      Processed 24000/49495 matches
      Processed 25000/49495 matches
      Processed 26000/49495 matches
      Processed 27000/49495 matches
      Processed 28000/49495 matches
      Processed 29000/49495 matches
      Processed 30000/49495 matches
      Processed 31000/49495 matches
      Processed 32000/49495 matches
      Processed 33000/49495 matches
      Processed 34000/49495 matches
      Processed 35000/49495 matches
      Processed 36000/49495 matches
      Processed 37000/49495 matches
      Processed 38000/49495 matches
      Processed 39000/49495 matches
      Processed 40000/49495 matches
      Processed 41000/49495 matches
      Processed 42000/49495 matches
      Processed 43000/49495 matches
      Processed 44000/49495 matches
      Processed 45000/49495 matches
      Processed 46000/49495 matches
      Processed 47000/49495 matches
      Processed 48000/49495 matches
      Processed 49000/49495 matches
      Processed 49495/49495 matches
      
      =========== FEATURE ENGINEERING SUMMARY ===========
      
      Number of matches : 49495
      Number of features: 31
      
      Feature Columns:
      
       - date
       - home_team
       - away_team
       - tournament
       - neutral
       - home_matches_played
       - away_matches_played
       - home_win_rate
       - away_win_rate
       - home_avg_goals
       - away_avg_goals
       - home_avg_conceded
       - away_avg_conceded
       - home_goal_difference
       - away_goal_difference
       - home_last5_win_rate
       - away_last5_win_rate
       - home_last5_avg_goals
       - away_last5_avg_goals
       - home_last5_avg_conceded
       - away_last5_avg_conceded
       - home_last5_goal_difference
       - away_last5_goal_difference
       - h2h_matches
       - h2h_home_team_win_rate
       - h2h_away_team_win_rate
       - h2h_draw_rate
       - h2h_home_avg_goals
       - h2h_away_avg_goals
       - h2h_goal_difference
       - target
      
      Preview:
      
              date home_team away_team  ... h2h_away_avg_goals  h2h_goal_difference    target
      0 1872-11-30  Scotland   England  ...           0.000000             0.000000      Draw
      1 1873-03-08   England  Scotland  ...           0.000000             0.000000  Home Win
      2 1874-03-07  Scotland   England  ...           2.000000            -1.000000  Home Win
      3 1875-03-06   England  Scotland  ...           1.333333             0.333333      Draw
      4 1876-03-04  Scotland   England  ...           1.750000            -0.250000  Home Win
      
      [5 rows x 31 columns]
      
      Feature dataset saved successfully at: 
      C:\xxxx\International_Football_Match_Outcome_Probability_Predictor\data\features\matches_features.csv
      
      Feature engineering completed successfully.


## 6. Choosing the model:

**File:** model.py

No single algorithm is universally best for every dataset. Each one has different strengths and assumptions, which allow their performance to be evaluated on the same engineered feature set. Therefore, this predictor project is evaluating several models with different learning strategies. These range from a simple linear model (Logistic Regression) to advanced ensemble methods (Random Forest, Gradient Boosting, and XGBoost). Then it compares and identifies the algorithm which provides the best balance of predictive accuracy, probability estimation, and generalization for football match outcome **(Home Win, Draw, Away Win)** prediction. This comparison also demonstrates how different machine learning approaches perform on the same engineered features.

Below I have included short explanations of all these algorithms about how these work, why one is chosen and what are the parameters to be set for each.

Note: For all these algorithms, a common **RANDOM_STATE (= 42**, need not be specific, any fixed value is fine) is used so that the algorithm makes the same random choices of training samples, producing the same model every time it is trained (assuming the data and parameters are unchanged). Without this, the final model may achieve slightly different accuracy which can make the experiments harder to compare. 

**A) Logistic Regression**

How it works:

Logistic Regression is not a regression algorithm, meaning it doesn’t predict continuous numbers; rather it is a linear classification algorithm that estimates the probability of each match outcome using the Softmax function. It starts with random weights and adjusts them to make the predicted probabilities match the actual labels as closely as possible. It does this by minimizing a loss function called binary cross-entropy (also called log loss), which penalizes confident wrong predictions much more heavily than confident right ones.

Once there is a probability, a threshold (usually 0.5) is picked and then:

    -	Probability ≥ 0.5 => predict class 1
    -	Probability < 0.5 => predict class 0

Why chosen:

    -	Simple and interpretable baseline model. 
    -	Produces class probabilities directly. 
    -	Fast to train and easy to understand. 
    -	Serves as a benchmark for comparing more complex models. 

Parameters:

    random_state=RANDOM_STATE : Ensures the model produces the same result every time it is trained on the same data and with the same code.  
    
    solver="lbfgs" : Optimization algorithm used to minimize the loss function. It stands for "Limited-memory Broyden–Fletcher–Goldfarb–Shanno". It is one of the most commonly used optimization algorithms for Logistic Regression in scikit-learn. Benefits of this include: Fast convergence, excellent for small and medium datasets and good default for multiclass classification.
    
    max_iter=3000 : Maximum optimization iterations to ensure convergence.


**B) Decision Tree**

How it works:

A decision tree predicts by asking a series of yes/no questions about the features, like a flowchart, until it reaches an answer.

E.g., to decide whether to play tennis today:

    is it raining?
        o	Yes -> Don't play
        o	No -> Is it windy?
              - Yes -> Don't play
              - No -> Play

That's a decision tree. Each internal node asks a question about one feature, each branch is an answer, and each leaf is a final prediction. At each step, the algorithm looks at all possible questions it could ask (for every feature, every possible split point) and picks the one that best separates the data into "pure" groups.

"Pure" means: after the split, each resulting group is mostly one class (mostly "play" or mostly "don't play"), rather than a 50/50 mess.

Building the Full Tree: 

The algorithm repeats this recursively: for each resulting group, it again looks for the best next split, and keeps going until:

      •	A group is pure (all one class), or
      •	It hits a stopping rule (e.g., max depth, minimum samples per leaf), this prevents the tree from growing so deep it just memorizes the training data (overfitting).

Making a Prediction: 

To predict on a new student, just walk down the tree answering the questions with that student's feature values until you land on a leaf that leaf's majority class is the prediction.

Why chosen:

    -	Easy to visualize and interpret (we can literally read the logic). 
    -	no need to scale features,
    -	handles nonlinear patterns naturally.
    -	Naturally handles interactions between features. 
    -	Provides insight into feature importance. 

    -	But prone to overfitting if grown too deep (memorizes noise) – that’s why Random Forest has come.


Parameters:

    random_state=RANDOM_STATE : Ensures reproducible tree construction.
    
    max_depth=10 : Limits the maximum tree depth to reduce overfitting.
    
    min_samples_split=5 : Minimum samples required before splitting a node.
    
    min_samples_leaf=2 : Ensures every leaf contains at least two samples for better generalization.


**C) Random Forest**

How it works:

First what does "Ensemble" Mean?

An ensemble method combines predictions from multiple models to get a better result than any single model alone.

Random Forest is an ensemble of many decision trees. Instead of building one tree (which is prone to overfitting and instability), we build many trees and let them vote. Each tree is trained on a random subset of the data (i.e., random sample of training data) and features (i.e., at each split in each tree, instead of considering all features to find the best question, the algorithm only considers a random subset of features). The final prediction is obtained by combining the predictions of all trees using majority voting (or averaged probabilities).

Why chosen:

    -	More accurate and robust than a single Decision Tree. 
    -	Reduces overfitting through ensemble learning. 
    -	Handles nonlinear relationships effectively. 
    -	Provides reliable feature importance estimates. 

Parameters:

    random_state=RANDOM_STATE : Produces reproducible results.
    n_estimators=300 : Number of trees in the forest. More trees generally improve stability.
    max_depth=15 : Restricts tree depth to prevent overfitting.
    min_samples_split=5 : Minimum samples required to split a node.
    min_samples_leaf=2 : Ensures every leaf contains at least two samples for better generalization.
    n_jobs=-1 : Uses all available CPU cores for parallel training.

**D) Gradient Boosting**

How it works:

Like Random Forests, Gradient Boosting is also an ensemble of decision trees. But it builds trees sequentially rather than independently. Each new tree is trained to correct the errors made by the previous trees. This iterative process gradually improves the model by minimizing the prediction loss.

Why chosen:

    -	Often achieves higher accuracy than Random Forest. 
    -	Learns complex relationships between features. 
    -	Produces well-calibrated probability estimates. 
    -	Strong baseline before using more advanced boosting algorithms. 
      
    -	But it is harder to parallelize as it is sequential.
    -	Also, it has higher overfitting risk if too many trees or the learning rate is too high – that’s why XGBoost has come.

Parameters:

    random_state=RANDOM_STATE : Ensures reproducible training
    n_estimators=200 : Number of boosting stages (trees).
    learning_rate=0.05 : Controls how much each tree contributes to the final model. Smaller values usually improve generalization but require more trees.
    max_depth=3 : Limits the depth of each individual decision tree (weak learner).


**E) XGBoost (Extreme Gradient Boosting)**

How it works:

XGBoost ("Extreme Gradient Boosting") is the same core idea as gradient boosting, but an optimized implementation of it. It builds trees sequentially, each correcting the previous one’s errors, but engineered to be faster, more accurate, and more resistant to overfitting. It's not a different algorithm philosophically; it's a highly optimized implementation with extra mathematical refinements. These improvements often lead to better predictive performance and faster training.

Why chosen:

    -	State-of-the-art algorithm for structured/tabular datasets. 
    -	Excellent predictive performance. 
    -	Handles complex feature interactions. 
    -	Efficient implementation with built-in regularization to reduce overfitting. 

Parameters:

    random_state=RANDOM_STATE : Ensures reproducible results.
    objective="multi:softprob" : Performs multiclass classification and outputs probabilities for each outcome.
    num_class=3 : Number of target classes (Home Win, Draw, Away Win).
    n_estimators=300 : Number of boosting trees.
    learning_rate=0.05 : Controls the contribution of each new tree to the ensemble.
    max_depth=6 : Maximum depth of each decision tree.
    subsample=0.8 : Randomly samples 80% of the training data for each tree to reduce overfitting.
    colsample_bytree=0.8 : Randomly samples 80% of the features when building each tree, improving diversity.
    eval_metric="mlogloss" : Uses multiclass logarithmic loss to evaluate training performance.
    tree_method="hist" : Uses the histogram-based tree construction algorithm for faster training on large datasets.

**Display Output:**

      Available Models:
      
       - Logistic Regression
       - Decision Tree
       - Random Forest
       - Gradient Boosting
       - XGBoost


## 7. Training pipeline:

**File:** train.py

Part 1: Data loading, feature preparation, train/test split;

Part 2: and then for each ML model: building the preprocessing pipeline, Training pipeline, training, and saving of the trained pipeline.

**Part 1:**

**A) load_features():** It loads the engineered features from matches_features.csv for the purpose of applying the transformation rules on the features so that the model can learn.

**B) prepare_dataset():** It converts the engineered dataset into the format expected by scikit-learn by separating:

      - input features (X): what the model learns from
      - and target labels (y): what the model should predict

Step 1: Original dataframe should not be changed, hence a copy of the dataframe is made for the changes to be done next.

Step 2: In the dataset, we have the target labels as {"Home Win", "Draw", "Away Win"}, which are texts and the ML algorithm can't learn from them. Therefore, mapping of target labels to numbers is required, which is defined next.

Step 3: Next the Label encoding happens, meaning the actual replacements of the string target labels with the mapping numbers are done.

Step 4: Then the target column is separated.

Step 5: Then the columns which are not required for training are dropped. 

      - target: it should not be seen during training, 
      - date: it is not relevant for training. It doesn't directly represent football knowledge.

Step 6: then it returns:

      - X : pandas.DataFrame -> input features
      - y : pandas.Series -> correct answers

Suppose the original dataset is:

      date		    home_team	  away_team	  home_win_rate	  away_win_rate	  target
      2022-11-23	Germany		  Japan		    0.71		        0.62		        Away Win
      2022-11-24	Brazil		  Serbia		  0.81		        0.53		        Home Win

After prepare_dataset():

X (Input Features):

      home_team	  away_team	  home_win_rate	  away_win_rate
      Germany		  Japan		    0.71		        0.62
      Brazil		  Serbia		  0.81		        0.53

y (Target Labels):

      target
      2
      0

**C) split_dataset():** It divides the dataset into training data and testing data. The purpose is to train the model on one set of matches and evaluate it on unseen matches to measure how well it generalizes.

      - inputs:The function receives: X = Input features, y = Target labels
      
      - split process: the dataset is split using the below settings:
      TEST_SIZE=0.20: meaning 80% of the matches = Training set, 20% of the matches = Testing set
      shuffle=False: because the order of the matches is important. The idea is to split the time-ordered dataset to train the model using past matches and perform the testing (get the predicttion) for recent or future matches.
      
      - returns:
      X_train: Features used for training.
      X_test: Features used for testing.
      y_train: Correct outcomes for the training matches.
      y_test: Correct outcomes for the testing matches.

**Part 2:**

**D) get_feature_category_details():** It separates the below feature categories and displays the details used for building the preprocessor (explained next) for specific model:

      - Categorical Features: ["home_team", "away_team", "tournament"] 
      - Numerical Features

**E) build_preprocessor():** data (more precisely few specific types of columns) transformation pipeline is defined here.

steps:

      a) For categorical features: as these contain non-numeric values, these need to be converted/encoded to numbers so that ML algorithms can understand. 
      
      - One-Hot encoding is applied here, which instead of assigning 1,2,3,4 to the teams like France, Germany, Brazil, Argentina respectively in colimns like home_team or away_team, i.e.,
      
      for:
      home_team	  away_team
      France		  Germany
      Brazil		  Argentina
      
      to:
      home_team	  away_team
      1		        2
      3		        4
      
      which incorrectly suggests Argentina > Brazil > Germany > France;
      
      creates separate binary columns to remove any false numerical ordering. e.g.,
      
      one match between France and Germany: (categories: France, Germany, Brazil)
      
      home_team	  away_team	  tournament
      France		  Germany		  World Cup
      
      gets encoded as below:
      
      home_France	  home_Germany	home_Brazil	  away_France	  away_Germany	away_Brazil	  World Cup	  Friendly
      1		          0		          0		          0		          1		          0		          1		        0
      
      That means, instead of using one column containing numbers, it creates one column for every category. So one original column becomes many columns.
      
      Note: One thing to notice here is that the number of columns/features increases because each category becomes its own binary feature. These make the model slower a bit, but modern algorithms are all designed to handle datasets with hundreds or even thousands of features.
      
      Why handle_unknown="ignore"?
      
      - so that the encoding during evaluation or prediction (cause the same preprocessing will be applied) doesn't fail for any unseen category (i.e., for a new team) appeared after the model was trained.
      
      b) For numerical features: as these contain numeric or boolean data, they don't need encoding, instead scaling is added.
      
      - StandardScaler(): for "Logistic Regression", which helps Logistic Regression converge much faster.
      - otherwise "passthrough": means no transformation is applied.
      
      c) Then ColumnTransformer combines all preprocessing rules into a single object.

Note: build_preprocessor() does not transform the data immediately. It creates a reusable set of instructions describing how each type of column should be transformed. Those instructions are later executed automatically inside the scikit-learn Pipeline whenever fit() or predict() is called.

**F) create_training_pipeline():** It combines all preprocessing steps and the machine learning model into a single object. And with this, I don't have to manually preprocess the data for each model (as my goal is to use multiple models for comparison).

Benefit of having pipeline during the training?

    When I call pipeline.fit(X_train, y_train), internally, scikit-learn performs:
    Preprocessor.fit(X_train) =>
    Preprocessor.transform(X_train) =>
    Model.fit(processed_X_train, y_train)
    
     - with one line, three operations happen automatically.

Benefit of having pipeline during the prediction?

    And when I call pipeline.predict(X_test), internally,
    Preprocessor.transform(X_test) =>
    Model.predict(processed_X_test)
    
     - preprocessing happens automatically before prediction.

**G) train_models():** for each model, it builds the preprocessor, builds the training pipeline and the actual training via fit pipeline:

        - get_models() from model.py to load all models: LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier, XGBClassifier
        - get_feature_category_details()
        - and then for each model:
              - builds the preprocessor specific to that model,
              - creates the training pipeline
              - fit pipeline
              - stores Trained Pipeline

**H) save_models():** Saves every trained pipeline.

**I) train_pipeline():** It combines all the steps of entire model training loop:

        - load_features(): Loads the dataset matches_features.csv
        - prepare_dataset()
        - split_dataset()
        - train_models()
        - Save all trained pipelines (.pkl)

**Display Output:**

      Loading feature dataset...
      Loaded 49495 matches
      
      Preparing dataset...
      Number of features : 29
      Number of samples  : 49495
      
      Splitting dataset...
      Training Matches : 39596
      Testing Matches  : 9899
      
      ========== Training Models ============
      
      Feature category details for building specific preprocessing steps...
      
      Categorical Features:
        home_team
        away_team
        tournament
      
      Numerical Features:
      26
      
      1: Training Logistic Regression...
      - Building preprocessing pipeline...
      - Building training pipeline...
      - Training completed for Logistic Regression.
      
      2: Training Decision Tree...
      - Building preprocessing pipeline...
      - Building training pipeline...
      - Training completed for Decision Tree.
      
      3: Training Random Forest...
      - Building preprocessing pipeline...
      - Building training pipeline...
      - Training completed for Random Forest.
      
      4: Training Gradient Boosting...
      - Building preprocessing pipeline...
      - Building training pipeline...
      - Training completed for Gradient Boosting.
      
      5: Training XGBoost...
      - Building preprocessing pipeline...
      - Building training pipeline...
      - Training completed for XGBoost.
      
      Saving Models...
      Logistic Regression Saved at C:\xxx\International_Football_Match_Outcome_Probability_Predictor\checkpoints\logistic_regression.pkl
      Decision Tree Saved at C:\xxx\International_Football_Match_Outcome_Probability_Predictor\checkpoints\decision_tree.pkl
      Random Forest Saved at C:\xxx\International_Football_Match_Outcome_Probability_Predictor\checkpoints\random_forest.pkl
      Gradient Boosting Saved at C:\xxx\International_Football_Match_Outcome_Probability_Predictor\checkpoints\gradient_boosting.pkl
      XGBoost Saved at C:\xxx\International_Football_Match_Outcome_Probability_Predictor\checkpoints\xgboost.pkl
      
      Training is finished successfully...

## 8. Model Evaluation:

This step evaluates the models on the test dataset by computing the metrices like accuracy, precision, recall, F1 Score, log loss etc.
It then chooses log loss as primary ranking metric (as the goal of this project is to find the match outcome probabilities and not to predict the exact correct outcome) to find the best performing model based on this metric.

**File:** evaluate.py

**Steps:**

The evaluation is done in below 2 phases:

      Phase 1: it includes the dataset preparation pipeline:
      
      A) load_features(): it loads engineered feature dataset and returns pandas.DataFrame.
      
      B) prepare_dataset(): It separates input features (X) and target labels (y) just the same way it was done during training phase.
           It returns:
           X : pandas.DataFrame -> input features
           y : pandas.Series -> correct answers
           
      C) split_dataset(): This function splits dataset into training and testing datasets. The important points here are:
           - I DO NOT shuffle.
           - Football prediction is a time-series style problem and hence,
               - Older matches are used for training.
               - Newer matches are used for testing.
           
           Returns:
             X_train
             X_test
             y_train
             y_test
             
      D) prepare_test_data(): This is the complete dataset preparation pipeline by combining the above functions.

      Note: No preprocessing is done here, as every saved pipeline (to be loaded in next phase) already contains (preprocessor + classifier).
      
      Phase 2: this phase includes the actual evaluation of the trained models and computation of performace metrices:
      
      E) load_models(): It loads all trained model pipelines from the models directory and returns a dictionary containing all trained pipelines.
      
      F) evaluate_model(): This evaluates a single trained model by predicting the most likely class, the class probabilities and by computing the standard evaluation metrices.
           Args taken:
             - model_name (str): Name of the model,
             - pipeline: Trained sklearn pipeline,
             - X_test (DataFrame): Test features,
             - y_test (Series): True labels,
           
           And it prints all these below evaluation metrices (for each model passed) and returns a dictionary of the computed metrices each time:
             - Accuracy: It measures the percentage of correctly predicted matches.
             - Precision: It tells when it predicts a class, what is the percentage of those predictions are correct on average across the three classes.
             - Recall:  it measures the ability of the model to correctly find all actual positive events (Hime Win, Away Win, Draw) out of all the times those events truly happened.
             - F1 Score: This is the balanced measure of precision and recall.
             - Log Loss: The "Log Loss" score indicates which model produces the best probability estimates.

      G) evaluate_all_models(): This function first loads the models (using load_models() function) and then evaluates each of the loaded models (by using evaluate_model() function). It rceives the evaluation metrices for each. It then sorts the models based on "Log Loss" metric in ascending order to indicate the best performing model as the primary goal of this predictor project is to predict match outcome probabilities rather than just the most likely class. 

        Why?
        - Accuracy only checks whether the predicted class is correct.
        - Log Loss evaluates how good the predicted probabilities are.
        
        For example, two models predicting a France vs Germany match:
        
        Model		  Home Win	Draw		Away Win	Correct?
        Model A		0.99		  0.005		0.005		  (Yes)
        Model B		0.55		  0.25		0.20		  (Yes)
        
        Both have the same accuracy, but Model B's probabilities are generally more realistic and better calibrated. Log Loss rewards models that assign sensible confidence levels rather than being overconfident.
                
      H) print_summary(): It displays all the computed metrices for all the models in tabular format as the model comparison overview.
      
      The winners based on each of the metric are as below:
      
      Metric	    Winner
      Accuracy	  Logistic Regression (58.22%)
      Precision	  Logistic Regression (50.56%)
      Recall	    Logistic Regression (50.04%)
      F1	        Logistic Regression (48%)
      Log Loss	  XGBoost (90.34%)

      But it prints XGBoost is the best model (based on Log Loss metric for the reason explained above). Because, although Logistic Regression achieved the highest classification accuracy (58.22%), XGBoost achieved the lowest Log Loss (0.9034). Since this project focuses on predicting match outcome probabilities rather than only the most likely outcome, Log Loss was selected as the primary evaluation metric. Therefore, XGBoost is chosen as the best-performing model.
      
       Conclusion from the evaluation: Overall, 58% accuracy is considered good, because for football prediction:
          - Random guessing (3 classes) ≈ 33%
          - Professional betting models often achieve 55–65% depending on the league and prediction task.
      
      So the model is significantly better than random guessing.
      
      But from the results, below points can be drawn:
            1) feature engineering is good. Even a simple Logistic Regression performs almost as well as XGBoost, suggesting the engineered features carry a lot of predictive information.
            2) There is probably more performance to gain from better features than from changing algorithms. Adding stronger features (e.g., Elo ratings, FIFA rankings etc.) is likely to produce a larger improvement than switching between these classifiers.
      
      Therefore, with hyperparameter tuning, richer features (such as Elo or FIFA rankings), and time-series-aware model selection, it can realistically improve its predictive performance, which I would consider as futute enhancements.

      I) save_evaluation_results(): This function saves the evaluation summary to a CSV file (evaluation_results.csv) in "results" folder. 
      
      This is useful for mainly two reasons:
         - to compare the models without rerunning the evaluation.
         - and to load the results later for plotting or analysis.
   
      J) evaluate_pipeline(): This function combines all the functions of phase 1 and 2 above, which is actually the entire evaluation pipeline.
          
**Display Output:**

      ======= Model Evaluation ============
      
      Loading feature dataset...
      Loaded 49495 matches
      
      Preparing dataset...
      Number of features : 29
      Number of samples  : 49495
      
      Splitting dataset...
      Training Matches : 39596
      Testing Matches  : 9899
      
      Loading trained models...
      Loaded: Decision Tree
      Loaded: Gradient Boosting
      Loaded: Logistic Regression
      Loaded: Random Forest
      Loaded: XGBoost
      
      Total Models Loaded: 5
      
      Evaluating All Models...
      
      Evaluating Decision Tree:
      Accuracy : 0.5407
      Precision: 0.4522
      Recall   : 0.4475
      F1 Score : 0.4195
      Log Loss : 1.8056
      
      Evaluating Gradient Boosting:
      Accuracy : 0.5764
      Precision: 0.4958
      Recall   : 0.4770
      F1 Score : 0.4335
      Log Loss : 0.9129
      
      Evaluating Logistic Regression:
      Accuracy : 0.5822
      Precision: 0.5056
      Recall   : 0.5004
      F1 Score : 0.4800
      Log Loss : 0.9048
      
      Evaluating Random Forest:
      Accuracy : 0.5682
      Precision: 0.3876
      Recall   : 0.4550
      F1 Score : 0.4020
      Log Loss : 0.9283
      
      Evaluating XGBoost:
      Accuracy : 0.5812
      Precision: 0.4949
      Recall   : 0.4864
      F1 Score : 0.4461
      Log Loss : 0.9034
      
      Model Comparison:
                       Model  Accuracy  Precision  Recall  F1 Score  Log Loss
      0              XGBoost    0.5812     0.4949  0.4864    0.4461    0.9034
      1  Logistic Regression    0.5822     0.5056  0.5004    0.4800    0.9048
      2    Gradient Boosting    0.5764     0.4958  0.4770    0.4335    0.9129
      3        Random Forest    0.5682     0.3876  0.4550    0.4020    0.9283
      4        Decision Tree    0.5407     0.4522  0.4475    0.4195    1.8056
      
      Best Model (Primary Ranking Metric = Log Loss)
      -------------------------------------------------
      Model    : XGBoost
      Accuracy : 0.5812
      F1 Score : 0.4461
      Log Loss : 0.9034
      
      Evaluation completed successfully.


## 8. Visualization:

This step plots various visualization figures from the evaluation summary for easy understanding of the model's performance.

**File:** visualization.py

**Figures:**

      A) Model comparison: plot_model_comparison()
      It plots model accuracy comparison bar chart for easy comparison of all the models.

<img width="2970" height="1766" alt="model_comparison" src="https://github.com/user-attachments/assets/254fa3d8-6319-4fe3-b828-d88f09c2799a" />

      The chart shows that all five models achieve fairly similar accuracy:

      XGBoost: 58.1% — highest
      Logistic Regression: 58.2% — actually marginally highest
      Gradient Boosting: 57.6%
      Random Forest: 56.8%
      Decision Tree: 54.1% — lowest
      
      Conclusion:
      Logistic Regression and XGBoost perform almost identically and are clearly the strongest models by accuracy. Decision Tree performs the weakest.
      
      However, since I have selected the best model based on Log Loss, the final choice of XGBoost is still justified: although Logistic Regression has a slightly higher accuracy (58.22% vs 58.12%), XGBoost has the lower Log Loss (0.9034 vs 0.9048) and therefore provides slightly better probability predictions.
      
      Below other visualizations are generated/examined only for the best model (XGBoost - the one with the lowest Log Loss):
      
      B) Confusion Matrix: plot_confusion_matrix()
      It plots the confusion matrix for standard classification evaluation.

<img width="1779" height="1770" alt="confusion_matrix" src="https://github.com/user-attachments/assets/974b1fd3-a448-4757-bf0f-691adc188565" />

      The matrix shows where XGBoost's predictions are correct and where it gets confused.

      Home Win: 4,050 correctly predicted out of 4,717 -> ~85.9% recall. Very good.
      Draw: Only 91 correctly predicted out of 2,310 -> ~3.9% recall. Very poor.
      Away Win: 1,612 correctly predicted out of 2,872 -> ~56.1% recall. Reasonable.
      
      Main observation: 
      The biggest problem is Draw prediction. Of the 2,310 actual draws, XGBoost predicted only 91 as Draw. Most were incorrectly classified as Home Win (1,532) or Away Win (687). This also explains the ROC result (shown below) for Draw (AUC = 0.606) — the model has difficulty distinguishing draws from wins.
      
      Overall conclusion:
      XGBoost is strong at identifying Home Wins, reasonably good at Away Wins, but performs poorly on Draws. This class imbalance/confusion is an important area for future improvement, perhaps through better draw-related features, class weighting, or probability calibration.

      C) ROC Curve: plot_roc_curve()
      It plots multiclass ROC curve (one-vs-rest) which shows probability discrimination. It measures how well the model can distinguish one outcome from the others.

<img width="2070" height="1765" alt="roc_curve" src="https://github.com/user-attachments/assets/31cda407-6e48-4675-bf88-eaf7aecf03d9" />

      From the figure we can see:
        Away Win: AUC = 0.780 (best)
        Home Win: AUC = 0.764 
        Draw: AUC = 0.606 (weakest)

        Hint: AUC (Area Under Curve): The probability that the model ranks a randomly chosen positive example higher than a randomly chosen negative example. For example, for Home Win (AUC = 0.764):
        Suppose we randomly pick:
        Match A -> actually Home Win
        Match B -> actually NOT Home Win
        There is approximately a 76.4% chance that XGBoost assigns a higher Home Win probability to Match A than Match B.
  
        Interpretation of each class:
        Home Win (AUC = 0.764): The blue curve is well above the diagonal.
        -> Good discrimination ability.
        -> The model can distinguish Home Wins from non-Home Wins fairly well.
        
        Away Win (AUC = 0.780): The green curve is the highest.
        -> Best performing class.
        -> The model is slightly better at identifying Away Wins than Home Wins.
        
        Draw (AUC = 0.606): The orange curve is much closer to the diagonal.
        -> Weak discrimination ability.
        -> The model struggles to separate Draws from non-Draws.
        
        This is very common in football prediction because draws are usually difficult to predict and often have fewer clear patterns than wins.
        
        Conclusion:
        XGBoost shows good discriminatory power for Home Wins (AUC = 0.764) and Away Wins (AUC = 0.780), indicating that the model can reliably distinguish these outcomes from other match results. However, Draws remain challenging to predict (AUC = 0.606), suggesting that additional features or alternative modeling approaches may be needed to improve draw prediction performance.
        
        Is this satisfactory?:
        For an international football match prediction project:
        AUC > 0.75  -> Good
        AUC 0.65-0.75 -> Reasonable
        AUC ~0.60 -> Weak but still better than random
        AUC = 0.50 -> Random guessing
        
        Home Win and Away Win performance are good, while Draw prediction is the main area for future improvement. This aligns with the earlier results where overall accuracy was around 58%, which is a realistic result for football outcome prediction.


      D) Calibration Curve: plot_calibration_curve()
          Args:
          model_name (str): Name of the model.
          y_test (Series): True labels.
          y_prob (ndarray): Predicted probabilities.
          
          It plots calibration curves for multiclass classification using a One-vs-Rest approach. This tells whether the predicted probabilities can be trusted. It basically shows whether predicted probabilities like 70% Home Win actually correspond to events happening about 70% of the time.

          Example:
          Suppose XGBoost predicts ~80% Home Win probability for 100 different matches in the test set:
          Match 1:  Team A vs Team B -> Home Win = 82%
          Match 2:  Team C vs Team D -> Home Win = 79%
          Match 3:  Team E vs Team F -> Home Win = 81%
          ...
          Match 100: Team X vs Team Y -> Home Win = 80%
          
          If the model is perfectly calibrated, then about 80 of those 100 matches should actually have been home wins. That's what the calibration curve compares: predicted probability (what the model predicted) versus observed frequency (what actually happened).
                    
          i) How to interpret the curve (in general):
          
          Suppose the graph for Home Win looks like this:

<img width="414" height="329" alt="image" src="https://github.com/user-attachments/assets/d485ac4d-9531-402c-8847-3873e7cb7b13" />

          The dashed diagonal represents perfect calibration.

          If the curve is close to the diagonal -> The probabilities are reliable.
          e.g., Predicted 70%  -> Actually happens about 70% of the time
          
          If the curve is below the diagonal -> The model is overconfident.
          e.g.,
          Predicts 90% -> Actually happens 70%
          => It is assigning probabilities that are too high.
          
          If the curve is above the diagonal -> The model is underconfident.
          e.g.,
          Predicts 60% -> Actually happens 80%
          => The model is more accurate than its confidence suggests.
          
          
          ii) The generated curve:

<img width="2370" height="1765" alt="calibration_curve" src="https://github.com/user-attachments/assets/6f4436e6-ed98-4999-b5e1-611a3432d70f" />

          iii) Interpretation of the generated curve:
          
          XGBoost calibration curve looks reasonably well calibrated overall. It tells:
          
          Home Win: The curve stays quite close to the diagonal = good calibration. XGBoost's Home Win probabilities are generally reliable.
          Away Win: The curve is mostly above the diagonal, especially around 0.5 – 0.75 = the model is somewhat underconfident for Away Win.
          Draw: The curve falls below the diagonal at higher probabilities = the model tends to be overconfident when predicting Draw.
          
          XGBoost demonstrates generally good probability calibration, particularly for Home Win and Away Win. However, the Draw probabilities show some overconfidence, indicating that the model's probability estimates for draws could be improved.

      
      E) Feature Importance: plot_feature_importance()
        Feature Importance (essential for tree-based models) indicates which features influenced the model's prediction most. This is only applicable to Random Forest, Gradient Boosting, XGBoost
        
        Below figure is for the best model again (XGBoost):

<img width="2964" height="1765" alt="feature_importance" src="https://github.com/user-attachments/assets/a8082d19-9efe-40e4-978c-08109dbb24cb" />

        Interpretations:
         - H2H goal difference is the most important feature. This suggests the historical goal difference between the two teams is highly useful for predicting the outcome.
         - Neutral venue is the second most important feature, indicating whether the match is played at a neutral location has a noticeable influence.
         - Away/home goal difference and H2H home-team win rate are also important.
         - Several tournament and team-specific features appear in the top 20, such as Germany, Brazil, Russia, and tournament types.
         - Recent form features such as home_last5_goal_difference also contribute, but less than the H2H and overall goal-difference features.
        
        Overall XGBoost relies most heavily on historical head-to-head performance and goal-difference statistics, while venue, recent form, team identity, and tournament context provide additional predictive information. One important point: feature importance does not tell us whether a feature increases or decreases the probability of a particular outcome. It only tells us how useful the feature was to the XGBoost model's decision-making.


      Also, two additional configuration flags are added:
      
      SAVE_FIGURES: this is set to "True" by default, which means that the visualization figures are saved (in "results" folder) every time these are generated.
      
      SHOW_FIGURES: this is set to "False" by default, which means that the visualization figures are not displayed or popped-up during a normal execution. This can be configured to "True" only during debugging for inspection.

