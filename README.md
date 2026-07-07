# International_Football_Match_Outcome_Probability_Predictor

A classical machine learning project that predicts the probability of possible match outcome (Home Win, Draw, Away Win) using historical football data of international matches. 

It focuses on structured tabular data and demonstrates the complete end-to-end machine learning workflow using Python and scikit-learn. The primary goal is to learn and apply classical machine learning concepts such as data preprocessing, feature engineering, model selection, hyperparameter tuning, and model evaluation.

## Objectives
  - Build a football match outcome prediction model using historical match data.
  - Compare multiple classical machine learning algorithms to identify the best-performing model.
  - Understand how feature engineering influences predictive performance.
  - Evaluate models using appropriate classification metrics.
  - Create a reusable prediction pipeline that can later be deployed as a web application or API.

## Choosing the dataset:
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


## Dataset details:
[https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017/data]

This dataset includes Total 49,393 results of international football matches.

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

shootouts.csv includes the following columns:

    date - date of the match
    home_team - the name of the home team
    away_team - the name of the away team
    winner - winner of the penalty-shootout
    first_shooter - the team that went first in the shootout

goalscorers.csv includes the following columns:

    date - date of the match
    home_team - the name of the home team
    away_team - the name of the away team
    team - name of the team scoring the goal
    scorer - name of the player scoring the goal
    own_goal - whether the goal was an own-goal
    penalty - whether the goal was a penalty

former_names.csv includes the following columns:

    current - name of the team as is used currently (or the last name if the team does not exist anymore)
    former - former name used by said team
    start_date - start date of when former name was used
    end_date - end date of when former name was used



