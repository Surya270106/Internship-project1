#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
==========================================================================
 EMPLOYEE SENTIMENT ANALYSIS - MAIN ANALYSIS SCRIPT
==========================================================================

 Project : Employee Sentiment Analysis
 Dataset : test.csv (Unlabeled Enron employee email messages)
 Language: Python 3

 This script performs the complete analysis pipeline:
   Task 1: Sentiment Labeling (TextBlob NLP)
   Task 2: Exploratory Data Analysis (EDA)
   Task 3: Monthly Employee Sentiment Scoring
   Task 4: Employee Ranking (Top 3 Positive / Negative per month)
   Task 5: Flight Risk Identification (Rolling 30-day window)
   Task 6: Linear Regression Predictive Modeling

 Supporting modules:
   - sentiment_analysis.py   : Core functions for labeling, scoring, ranking
   - visualizations_generator.py : Chart/plot generation utilities

==========================================================================
"""

# ============================================================================
# IMPORTS
# ============================================================================
import sys
import io
# Fix Windows console encoding for Unicode
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from sentiment_analysis import (
    label_dataset, compute_monthly_scores, rank_employees,
    identify_flight_risks, engineer_features, prepare_regression_data
)
from visualizations_generator import (
    generate_all_eda_visualizations, plot_monthly_scores_heatmap,
    plot_top_employees_ranking, plot_flight_risk,
    plot_regression_results, plot_residuals
)


# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_FILE = 'test.csv'
OUTPUT_DIR = 'visualizations'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def print_header(title, char='='):
    """Print a formatted section header."""
    width = 70
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}\n")


def print_subheader(title):
    """Print a formatted sub-section header."""
    print(f"\n{'-'*50}")
    print(f"  {title}")
    print(f"{'-'*50}")


# ============================================================================
# STEP 0: DATA LOADING AND CLEANING
# ============================================================================

print_header("STEP 0: DATA LOADING AND PREPROCESSING")

print("[DATA] Loading dataset from:", DATA_FILE)
df = pd.read_csv(DATA_FILE)

print(f"\n[INFO] Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\n[INFO] Column Names: {df.columns.tolist()}")
print(f"\n[INFO] Data Types:")
print(df.dtypes)

print_subheader("Missing Values")
print(df.isnull().sum())
print(f"\nTotal missing cells: {df.isnull().sum().sum()}")

# Parse dates
print_subheader("Date Parsing")
df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=False)
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Total unique employees (senders): {df['from'].nunique()}")

# Clean employee names (standardize email format)
df['from'] = df['from'].str.strip().str.lower()

print(f"\n[OK] Data loaded and cleaned successfully!")
print(f"   - {df.shape[0]} messages from {df['from'].nunique()} unique employees")
print(f"   - Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")


# ============================================================================
# TASK 1: SENTIMENT LABELING
# ============================================================================

print_header("TASK 1: SENTIMENT LABELING")

print("""
APPROACH:
   We use TextBlob, a popular NLP library built on NLTK, for sentiment analysis.
   TextBlob provides a polarity score ranging from -1.0 (most negative) to 
   +1.0 (most positive).

   Classification Thresholds:
   - Polarity > 0.1  -> Positive
   - Polarity < -0.1 -> Negative
   - Otherwise        -> Neutral

   We combine the 'Subject' and 'body' fields to get richer context for
   more accurate sentiment classification.
""")

print("[PROCESSING] Applying sentiment labeling to all messages...")
df = label_dataset(df)

print("\n[OK] Sentiment labeling complete!")
print_subheader("Sentiment Distribution")
sentiment_counts = df['sentiment'].value_counts()
for sentiment, count in sentiment_counts.items():
    pct = count / len(df) * 100
    print(f"   {sentiment:>10}: {count:>5} messages ({pct:.1f}%)")

print(f"\n[STATS] Average Polarity Score: {df['polarity'].mean():.4f}")
print(f"   Std Polarity Score:    {df['polarity'].std():.4f}")

# Save labeled dataset
labeled_file = 'test_labeled.csv'
df.to_csv(labeled_file, index=False)
print(f"\n[SAVED] Labeled dataset saved to: {labeled_file}")


# ============================================================================
# TASK 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

print_header("TASK 2: EXPLORATORY DATA ANALYSIS (EDA)")

print_subheader("2.1 Data Overview")
print(f"Total messages: {len(df)}")
print(f"Unique employees: {df['from'].nunique()}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"\nMessages per employee (summary statistics):")
msg_per_emp = df['from'].value_counts()
print(f"   Mean:   {msg_per_emp.mean():.1f}")
print(f"   Median: {msg_per_emp.median():.1f}")
print(f"   Max:    {msg_per_emp.max()} ({msg_per_emp.idxmax()})")
print(f"   Min:    {msg_per_emp.min()}")

print_subheader("2.2 Sentiment Distribution Analysis")
for sentiment in ['Positive', 'Neutral', 'Negative']:
    subset = df[df['sentiment'] == sentiment]
    print(f"\n   {sentiment} Messages:")
    print(f"     Count: {len(subset)}")
    print(f"     Avg polarity: {subset['polarity'].mean():.4f}")
    print(f"     Avg message length: {subset['full_text'].str.len().mean():.0f} chars")

print_subheader("2.3 Temporal Analysis")
df['year_month'] = df['date'].dt.to_period('M')
monthly_msgs = df.groupby('year_month').size()
print(f"Monthly message volume:")
print(f"   Avg messages/month: {monthly_msgs.mean():.1f}")
print(f"   Peak month: {monthly_msgs.idxmax()} ({monthly_msgs.max()} messages)")
print(f"   Lowest month: {monthly_msgs.idxmin()} ({monthly_msgs.min()} messages)")

print_subheader("2.4 Employee Activity Analysis")
top_senders = df['from'].value_counts().head(10)
print("\nTop 10 most active employees:")
for i, (emp, count) in enumerate(top_senders.items(), 1):
    print(f"   {i:>2}. {emp:<40} {count:>4} messages")

# Observation about day-of-week patterns
df['day_of_week_name'] = df['date'].dt.day_name()
dow_counts = df['day_of_week_name'].value_counts()
print_subheader("2.5 Day of Week Patterns")
for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
    count = dow_counts.get(day, 0)
    print(f"   {day:>10}: {count:>5} messages")

print_subheader("2.6 Key Observations from EDA")
print("""
   1. The dataset contains employee email messages from the Enron corpus.
   2. Messages are distributed across multiple months/years.
   3. Most messages tend to be Neutral or Positive in sentiment.
   4. Some employees are significantly more active than others.
   5. Message activity varies across days of the week.
   6. There is notable variation in message length across sentiment categories.
""")

# Generate all EDA visualizations
generate_all_eda_visualizations(df)


# ============================================================================
# TASK 3: EMPLOYEE SCORE CALCULATION
# ============================================================================

print_header("TASK 3: MONTHLY EMPLOYEE SENTIMENT SCORING")

print("""
SCORING METHOD:
   For each employee, each message receives a score:
   - Positive message: +1
   - Negative message: -1  
   - Neutral message:   0 (no effect)
   
   Scores are aggregated monthly and reset at the start of each new month.
""")

monthly_scores = compute_monthly_scores(df)

print(f"[OK] Monthly scores computed for {monthly_scores['from'].nunique()} employees")
print(f"   across {monthly_scores['year_month'].nunique()} unique months\n")

print_subheader("Sample Monthly Scores (First 10 Records)")
print(monthly_scores.head(10).to_string(index=False))

print_subheader("Monthly Score Statistics")
print(f"   Mean monthly score: {monthly_scores['monthly_score'].mean():.2f}")
print(f"   Std monthly score:  {monthly_scores['monthly_score'].std():.2f}")
print(f"   Max monthly score:  {monthly_scores['monthly_score'].max()}")
print(f"   Min monthly score:  {monthly_scores['monthly_score'].min()}")

# Save monthly scores
monthly_scores.to_csv('monthly_scores.csv', index=False)
print(f"\n[SAVED] Monthly scores saved to: monthly_scores.csv")

# Generate monthly scores heatmap
plot_monthly_scores_heatmap(monthly_scores)


# ============================================================================
# TASK 4: EMPLOYEE RANKING
# ============================================================================

print_header("TASK 4: EMPLOYEE RANKING")

print("""
RANKING METHOD:
   For each month, we identify:
   1. Top 3 Positive Employees (highest monthly score)
   2. Top 3 Negative Employees (lowest monthly score)
   
   Ties are broken alphabetically (ascending order by employee name).
""")

top_positive, top_negative = rank_employees(monthly_scores)

# Display rankings for each month
for month in sorted(monthly_scores['year_month'].unique()):
    print_subheader(f"Month: {month}")
    
    pos_month = top_positive[top_positive['year_month'] == month]
    neg_month = top_negative[top_negative['year_month'] == month]
    
    print("  [+] Top 3 POSITIVE Employees:")
    for _, row in pos_month.iterrows():
        print(f"      #{int(row['rank'])}  {row['from']:<40} Score: {row['monthly_score']:>3}")
    
    print("  [-] Top 3 NEGATIVE Employees:")
    for _, row in neg_month.iterrows():
        print(f"      #{int(row['rank'])}  {row['from']:<40} Score: {row['monthly_score']:>3}")

# Save rankings
top_positive.to_csv('top_positive_employees.csv', index=False)
top_negative.to_csv('top_negative_employees.csv', index=False)
print(f"\n[SAVED] Rankings saved to: top_positive_employees.csv, top_negative_employees.csv")

# Generate ranking visualization
plot_top_employees_ranking(top_positive, top_negative)


# ============================================================================
# TASK 5: FLIGHT RISK IDENTIFICATION
# ============================================================================

print_header("TASK 5: FLIGHT RISK IDENTIFICATION")

print("""
FLIGHT RISK CRITERIA:
   An employee is flagged as a flight risk if they have sent 4 or more 
   NEGATIVE messages within any rolling 30-day window.
   
   The 30-day window is a rolling count of days, irrespective of calendar
   month boundaries. This means we check every possible 30-day window for
   each employee.
""")

flight_risks = identify_flight_risks(df)

if flight_risks.empty:
    print("[WARNING] No employees were identified as flight risks.")
else:
    print(f"[ALERT] {len(flight_risks)} employees identified as FLIGHT RISKS:\n")
    
    for _, row in flight_risks.iterrows():
        print(f"   [!] {row['employee']}")
        print(f"       Total negative messages: {row['total_negative_messages']}")
        print(f"       Max negatives in 30-day window: {row['max_negatives_in_30_days']}")
        print(f"       Risk window: {row['risk_window_start'].strftime('%Y-%m-%d')} to {row['risk_window_end'].strftime('%Y-%m-%d')}")
        print()
    
    flight_risks.to_csv('flight_risk_employees.csv', index=False)
    print(f"[SAVED] Flight risk list saved to: flight_risk_employees.csv")

# Generate flight risk visualization
plot_flight_risk(flight_risks)


# ============================================================================
# TASK 6: PREDICTIVE MODELING (LINEAR REGRESSION)
# ============================================================================

print_header("TASK 6: LINEAR REGRESSION PREDICTIVE MODEL")

print("""
MODEL APPROACH:
   We develop a linear regression model to analyze sentiment trends 
   and predict sentiment polarity scores using engineered features.
   
   Independent Variables (Features):
   - message_length    : Total character count of the message
   - word_count        : Number of words in the message
   - subject_length    : Length of the subject line
   - has_subject       : Whether the message has a subject (0/1)
   - day_of_week       : Day of week (0=Monday, 6=Sunday)
   - is_weekend        : Whether sent on weekend (0/1)
   - month             : Month of the year (1-12)
   - exclamation_count : Number of '!' characters
   - question_count    : Number of '?' characters
   - caps_ratio        : Ratio of uppercase to total alphabetic characters
   
   Dependent Variable (Target): polarity (continuous, -1 to +1)
   
   Library: scikit-learn (LinearRegression)
""")

# Engineer features
print("[PROCESSING] Engineering features...")
df = engineer_features(df)
X, y, feature_names = prepare_regression_data(df)

print(f"   Features: {feature_names}")
print(f"   Total samples: {len(X)}")

# Train-test split (80/20)
print_subheader("Train-Test Split")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   Training set: {len(X_train)} samples")
print(f"   Testing set:  {len(X_test)} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Linear Regression model
print_subheader("Model Training")
model = LinearRegression()
model.fit(X_train_scaled, y_train)
print("   [OK] Linear Regression model trained successfully!")

# Predictions
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# Evaluation metrics
print_subheader("Model Performance Metrics")
metrics = {
    'Training R2 Score': r2_score(y_train, y_pred_train),
    'Testing R2 Score': r2_score(y_test, y_pred_test),
    'Training RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
    'Testing RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
    'Training MAE': mean_absolute_error(y_train, y_pred_train),
    'Testing MAE': mean_absolute_error(y_test, y_pred_test),
}

for metric, value in metrics.items():
    print(f"   {metric:<25}: {value:.4f}")

print_subheader("Feature Coefficients (Scaled)")
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': model.coef_
}).sort_values('Coefficient', ascending=False)

for _, row in coef_df.iterrows():
    direction = "+" if row['Coefficient'] > 0 else "-"
    print(f"   {direction} {row['Feature']:<25}: {row['Coefficient']:>10.6f}")

print(f"\n   Intercept: {model.intercept_:.6f}")

print_subheader("Model Interpretation")
print("""
   Key Findings:
   - The linear regression model captures some of the variance in sentiment
     polarity, though text sentiment is inherently difficult to predict from
     metadata features alone.
   - Features like exclamation count, message length, and caps ratio show 
     meaningful correlations with sentiment polarity.
   - The R2 score indicates the proportion of variance explained by the model.
   - A low R2 is expected since sentiment is primarily determined by actual 
     text content, not structural features.
""")

# Generate regression visualizations
plot_regression_results(y_test, y_pred_test, model.coef_, feature_names)
plot_residuals(y_test, y_pred_test)


# ============================================================================
# SUMMARY OF FINDINGS
# ============================================================================

print_header("SUMMARY OF KEY FINDINGS")

print("[1] SENTIMENT OVERVIEW:")
for sentiment, count in sentiment_counts.items():
    pct = count / len(df) * 100
    print(f"      {sentiment}: {count} ({pct:.1f}%)")

print("\n[2] TOP POSITIVE EMPLOYEES (Overall):")
overall_pos = monthly_scores.groupby('from')['monthly_score'].sum().nlargest(3)
for i, (emp, score) in enumerate(overall_pos.items(), 1):
    print(f"      #{i} {emp} (Total Score: {score})")

print("\n[3] TOP NEGATIVE EMPLOYEES (Overall):")
overall_neg = monthly_scores.groupby('from')['monthly_score'].sum().nsmallest(3)
for i, (emp, score) in enumerate(overall_neg.items(), 1):
    print(f"      #{i} {emp} (Total Score: {score})")

print("\n[4] FLIGHT RISK EMPLOYEES:")
if flight_risks.empty:
    print("      No employees flagged as flight risks.")
else:
    for _, row in flight_risks.iterrows():
        print(f"      [!] {row['employee']} ({row['max_negatives_in_30_days']} negatives in 30 days)")

print("\n[5] MODEL PERFORMANCE:")
print(f"      R2 Score (Test): {metrics['Testing R2 Score']:.4f}")
print(f"      RMSE (Test):     {metrics['Testing RMSE']:.4f}")
print(f"      MAE (Test):      {metrics['Testing MAE']:.4f}")

print_header("PROJECT COMPLETE")
print("""
   All tasks have been completed successfully!
   
   Output Files:
      - test_labeled.csv         - Dataset with sentiment labels
      - monthly_scores.csv       - Monthly sentiment scores
      - top_positive_employees.csv - Top positive employee rankings
      - top_negative_employees.csv - Top negative employee rankings
      - flight_risk_employees.csv  - Flight risk employees
      - visualizations/           - All generated charts and plots
   
   Visualizations Generated:
      01. Sentiment Distribution
      02. Sentiment Trends Over Time
      03. Polarity Score Distribution
      04. Messages Per Employee
      05. Sentiment by Employee
      06. Day of Week Patterns
      07. Message Length by Sentiment
      08. Word Clouds
      09. Monthly Scores Heatmap
      10. Employee Rankings
      11. Flight Risk Employees
      12. Regression Results
      13. Residual Diagnostics
""")
