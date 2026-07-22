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
    
    solver="saga" : Optimization algorithm used to minimize the loss function. It is well suited for larger, sparse feature matrices produced by one-hot encoding. Also it is good for multiclass problems (to classify all three outcomes simultaneously (Home Win, Draw, Away Win)). It scales well and supports multinomial logistic regression. 
    
    max_iter=1000 : Maximum optimization iterations to ensure convergence.
    
    n_jobs=-1 : This allows parallel computation where supported by the solver, making training faster on multicore CPUs.


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


## 7. Training pipeline:

**File:** train.py

Part 1: Data loading, feature preparation, train/test split, and building the preprocessing pipeline.

Part 2: Training all models, saving them, and the training pipeline.

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

**D) build_preprocessor():** data (more precisely few specific types of columns) transformation pipeline is defined here.

steps:

      a) categorical features are separated: as these contain non-numeric values, these need to be converted/encoded to numbers so that ML algorithms can understand. 
      
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
      
      b) numerical features are separated: these contain numeric or boolean data, so they don't need encoding.
      
      - passthrough: means no transformation is applied.
      
      c) Then ColumnTransformer combines all preprocessing rules into a single object.

Note: build_preprocessor() does not transform the data immediately. It creates a reusable set of instructions describing how each type of column should be transformed. Those instructions are later executed automatically inside the scikit-learn Pipeline whenever fit() or predict() is called.

**E) create_pipeline():** It combines all preprocessing steps and the machine learning model into a single object. And with this, I don't have to manually preprocess the data for each model (as my goal is to use multiple models for comparison).

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

**F) save_preprocessor():** In this step, the preprocessor is saved so that it can be used during prediction too.

**G) create_model_preprocessing_pipeline():** It combines all the steps of model preprocessing.

