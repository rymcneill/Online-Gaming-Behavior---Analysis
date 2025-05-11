# Online Gaming Behavior Analysis

This repository contains the Python code and outputs for analyzing online gaming behavior data to predict player engagement levels. The analysis includes data preprocessing, exploratory data analysis (EDA), feature engineering, model building (Logistic Regression, Random Forest, XGBoost), model evaluation, and visualization of results.

## Data Source

The dataset used for this analysis is `online_gaming_behavior_dataset.csv`, which contains information about player activity, demographics, and in-game behavior. (If publicly available, add a link here). Key features include playtime, session frequency, player level, achievements, in-game purchases, and more.

## Methodology

The analysis pipeline involves the following key steps:

1.  **Data Loading and Cleaning:** Handling missing values and outliers.
2.  **Data Preprocessing:** Encoding categorical variables and normalizing numerical features.
3.  **Feature Engineering:** Creating new relevant features, such as total playtime per week.
4.  **Feature Selection:** Identifying the most important features for predicting engagement.
5.  **Model Building:** Training and tuning three classification models: Logistic Regression, Random Forest, and XGBoost.
6.  **Model Evaluation:** Assessing model performance using cross-validation, confusion matrices, and classification reports.
7.  **Results Visualization:** Creating plots to understand data distributions, feature correlations, and model performance.

## Key Findings

- Random Forest achieved the highest prediction accuracy, XGBoost is a close second.
- The feature TotalPlayMinutesPerWeek has the highest correlated feature to a player's engagement level.
- The features that have the lowest correlation to the a player's engagement level is gender.

## Project Structure

The project directory is organized as follows:

Online-Gaming-Behavior-Analysis/

├── data/

│   └── online_gaming_behavior_dataset.csv  # The raw dataset

├── notebooks/

│   └── analysis.ipynb                      # The main Jupyter Notebook containing the analysis pipeline

├── outputs/

│   ├── figures/                          # Directory to store generated plots

│   │   ├── confusion_matrix_logistic_regression.png

│   │   ├── confusion_matrix_random_forest.png

│   │   ├── confusion_matrix_xgboost.png

│   │   ├── model_accuracy_comparison.png

│   │   ├── model_f1_score_comparison.png

│   │   ├── xgboost_feature_importance.png

│   │   ├── engagement_level_distribution.png

│   │   ├── feature_correlation_heatmap.png

│   │   ├── pairwise_plot_key_features.png

│   │   ├── play_minutes_by_engagement_level.png

│   │   ├── distribution_PlayTimeHours.png

│   │   ├── distribution_SessionsPerWeek.png

│   │   ├── distribution_AvgSessionDurationMinutes.png

│   │   └── distribution_AchievementsUnlocked.png

│   └── results/                          # Directory to store textual results

│       ├── cross_validation_scores.txt

│       ├── logistic_regression_report.txt

│       ├── random_forest_report.txt

│       ├── xgboost_report.txt

│       └── model_scores.csv

├── scr/

│   └── main_pipeline.py                    # Main pipeline, containing the entire Python script

├── README.md                             # This file

└── requirements.txt                      # List of project dependencies

