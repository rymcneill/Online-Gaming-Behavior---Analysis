# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

# Set seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Load the data
df = pd.read_csv("data/online_gaming_behavior_dataset.csv")

# Check for missing values
print(df.isnull().sum())
# Fill missing values
# Age: Fill with median to avoid outlier skew
# AvgSessionDurationMinutes: Fill with mean to retain overall distribution
df['Age'].fillna(df['Age'].median(), inplace=True)
df['AvgSessionDurationMinutes'].fillna(df['AvgSessionDurationMinutes'].mean(), inplace=True)

# Handle outliers using Z-score for numeric columns
num_cols = ['PlayTimeHours', 'PlayerLevel', 'AchievementsUnlocked']
df = df[(np.abs(zscore(df[num_cols])) < 3).all(axis=1)]

# Encode categorical variables
categorical_cols = ['Gender', 'Location', 'GameGenre', 'GameDifficulty']
df = pd.get_dummies(df, columns=categorical_cols)

# Encode target variable
engagement_map = {'Low': 0, 'Medium': 1, 'High': 2}
df['EngagementLevel'] = df['EngagementLevel'].map(engagement_map)

# Normalize selected features
scaler = MinMaxScaler()
scaled_features = ['PlayTimeHours', 'SessionsPerWeek', 'AvgSessionDurationMinutes', 'AchievementsUnlocked']
df[scaled_features] = scaler.fit_transform(df[scaled_features])

# Feature Engineering
# Create new feature: Total playtime per week
df['TotalPlayMinutesPerWeek'] = df['SessionsPerWeek'] * df['AvgSessionDurationMinutes']

# Feature Selection using correlation
target_corr = df.corr()['EngagementLevel'].sort_values(ascending=False)
print("\nTop correlated features:\n", target_corr.head(10))

# Select top features manually or based on importance (can use RF importance later)
selected_features = ['TotalPlayMinutesPerWeek', 'PlayerLevel', 'AchievementsUnlocked', 'PlayTimeHours', 'InGamePurchases']

X = df[selected_features]
y = df['EngagementLevel']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# Baseline Model Implementation
# Logistic Regression
log_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

# Random Forest with GridSearchCV
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_grid = GridSearchCV(RandomForestClassifier(random_state=RANDOM_STATE),
                       param_grid=rf_param_grid,
                       cv=5,
                       scoring='accuracy',
                       n_jobs=-1)
rf_grid.fit(X_train, y_train)
best_rf_model = rf_grid.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)

# XGBoost with GridSearchCV
xgb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'learning_rate': [0.1, 0.3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb_grid = GridSearchCV(XGBClassifier(eval_metric='mlogloss', random_state=RANDOM_STATE),
                        param_grid=xgb_param_grid,
                        cv=5,
                        scoring='accuracy',
                        n_jobs=-1)
xgb_grid.fit(X_train, y_train)
best_xgb_model = xgb_grid.best_estimator_
y_pred_xgb = best_xgb_model.predict(X_test)

# Cross-validation
log_cv_scores = cross_val_score(log_model, X, y, cv=5)
rf_cv_scores = cross_val_score(best_rf_model, X, y, cv=5)
xgb_cv_scores = cross_val_score(best_xgb_model, X, y, cv=5)

with open('outputs/results/cross_validation_scores.txt', 'w') as f:
    f.write(f"Logistic Regression CV Score: {log_cv_scores.mean():.4f}\n")
    f.write(f"Random Forest CV Score: {rf_cv_scores.mean():.4f}\n")
    f.write(f"XGBoost CV Score: {xgb_cv_scores.mean():.4f}\n")

print("\nCross-Validation Scores:")
print(f"Logistic Regression: {log_cv_scores.mean():.4f}")
print(f"Random Forest: {rf_cv_scores.mean():.4f}")
print(f"XGBoost: {xgb_cv_scores.mean():.4f}")

# Confusion Matrices
cm_log = confusion_matrix(y_test, y_pred_log)
sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues')
plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig('outputs/figures/logistic_regression_cm.png')
plt.show()

cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens')
plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig('outputs/figures/random_forest_cm.png')
plt.show()

cm_xgb = confusion_matrix(y_test, y_pred_xgb)
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Oranges')
plt.title("XGBoost - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig('outputs/figures/xgboost_cm.png')
plt.show()

log_report = classification_report(y_test, y_pred_log)
rf_report = classification_report(y_test, y_pred_rf)
xgb_report = classification_report(y_test, y_pred_xgb)

print("\nLogistic Regression Report:")
print(log_report)

print("\nRandom Forest Report:")
print(rf_report)

print("\nXGBoost Report:")
print(xgb_report)

# Save classification reports to files
with open('outputs/results/logistic_regression_report.txt', 'w') as f:
    f.write(log_report)
with open('outputs/results/random_forest_report.txt', 'w') as f:
    f.write(rf_report)
with open('outputs/results/xgboost_report.txt', 'w') as f:
    f.write(xgb_report)
# Summary Table
model_scores = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_log),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_xgb)
    ],
    'F1 Score': [
        f1_score(y_test, y_pred_log, average='macro'),
        f1_score(y_test, y_pred_rf, average='macro'),
        f1_score(y_test, y_pred_xgb, average='macro')
    ]
})
print("\nModel Scores:")
print(model_scores)

#adding models to files
model_scores.to_csv('outputs/results/model_scores.csv', index=False)


# Visual Comparison
sns.barplot(x='Model', y='Accuracy', data=model_scores)
plt.title("Model Accuracy Comparison")
plt.savefig('outputs/figures/model_accuracy_comparison.png')
plt.show()

sns.barplot(x='Model', y='F1 Score', data=model_scores)
plt.title("Model F1 Score Comparison")
plt.savefig('outputs/figures/model_f1_score_comparison.png')  # Save the plot
plt.show()

#Feature Importance (XGBoost)
importances = best_xgb_model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot - Feature Importance (XGBoost)
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
plt.title('XGBoost Feature Importances')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('outputs/figures/xgboost_feature_importance.png')  # Save the plot
plt.show()

#Target Variable Distribution
sns.countplot(x='EngagementLevel', data=df)
plt.title("Distribution of Engagement Levels")
plt.xlabel("Engagement Level")
plt.ylabel("Count")
plt.savefig('outputs/figures/engagement_level_distribution.png')  # Save the plot
plt.show()

#Correlation Heatmap (numeric features)
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.savefig('outputs/figures/feature_correlation_heatmap.png')  # Save the plot
plt.show()

#Pairplot of Key Features
selected_numeric = ['TotalPlayMinutesPerWeek', 'PlayerLevel', 'AchievementsUnlocked', 'PlayTimeHours', 'EngagementLevel']
sns.pairplot(df[selected_numeric], hue='EngagementLevel', diag_kind='kde')
plt.suptitle("Pairwise Plot of Key Features", y=1.02)
plt.savefig('outputs/figures/pairwise_plot_key_features.png')  # Save the plot
plt.show()

#Boxplot: Feature Distributions by Engagement
plt.figure(figsize=(10, 6))
sns.boxplot(x='EngagementLevel', y='TotalPlayMinutesPerWeek', data=df)
plt.title("Total Play Minutes Per Week by Engagement Level")
plt.savefig('outputs/figures/play_minutes_by_engagement_level.png')  # Save the plot
plt.show()

#Distribution Plot of Normalized Features
for feature in scaled_features:
    sns.histplot(df[feature], kde=True)
    plt.title(f"Distribution of {feature}")
    plt.savefig(f'outputs/figures/distribution_{feature}.png')  # Save the plot
    plt.show()
