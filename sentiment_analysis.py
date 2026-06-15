"""
sentiment_analysis.py
=====================
Supporting module for Employee Sentiment Analysis project.
Contains utility functions for sentiment labeling, scoring, ranking,
flight risk identification, and predictive modeling.

Author: Employee Sentiment Analysis Project
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# TASK 1: SENTIMENT LABELING
# ============================================================================

def get_sentiment_label(text):
    """
    Analyze sentiment of a given text using TextBlob.
    
    Approach:
    - Uses TextBlob's built-in sentiment polarity analyzer.
    - Polarity ranges from -1 (most negative) to +1 (most positive).
    - Classification thresholds:
        * Polarity > 0.1  → Positive
        * Polarity < -0.1 → Negative
        * Otherwise        → Neutral
    
    Parameters:
        text (str): The text to analyze.
    
    Returns:
        tuple: (sentiment_label, polarity_score)
    """
    if pd.isna(text) or str(text).strip() == '':
        return 'Neutral', 0.0
    
    try:
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            return 'Positive', polarity
        elif polarity < -0.1:
            return 'Negative', polarity
        else:
            return 'Neutral', polarity
    except Exception:
        return 'Neutral', 0.0


def label_dataset(df):
    """
    Apply sentiment labeling to the entire dataset.
    
    Combines 'Subject' and 'body' fields for richer context,
    then applies TextBlob sentiment analysis.
    
    Parameters:
        df (pd.DataFrame): DataFrame with 'Subject' and 'body' columns.
    
    Returns:
        pd.DataFrame: DataFrame with added 'sentiment' and 'polarity' columns.
    """
    # Combine subject and body for more context
    df['full_text'] = df['Subject'].fillna('') + ' ' + df['body'].fillna('')
    
    # Apply sentiment analysis
    results = df['full_text'].apply(lambda x: get_sentiment_label(x))
    df['sentiment'] = results.apply(lambda x: x[0])
    df['polarity'] = results.apply(lambda x: x[1])
    
    return df


# ============================================================================
# TASK 3: EMPLOYEE SCORE CALCULATION
# ============================================================================

def calculate_sentiment_score(sentiment):
    """
    Convert sentiment label to a numerical score.
    
    Scoring:
        Positive → +1
        Negative → -1
        Neutral  →  0
    
    Parameters:
        sentiment (str): Sentiment label.
    
    Returns:
        int: Numerical score.
    """
    if sentiment == 'Positive':
        return 1
    elif sentiment == 'Negative':
        return -1
    else:
        return 0


def compute_monthly_scores(df):
    """
    Compute monthly sentiment scores for each employee.
    
    Groups messages by employee (from) and month (year_month),
    then sums the sentiment scores for each group.
    
    Parameters:
        df (pd.DataFrame): DataFrame with 'from', 'date', and 'sentiment' columns.
    
    Returns:
        pd.DataFrame: Monthly scores with columns [from, year_month, monthly_score,
                       positive_count, negative_count, neutral_count, total_messages].
    """
    df['sentiment_score'] = df['sentiment'].apply(calculate_sentiment_score)
    df['year_month'] = df['date'].dt.to_period('M')
    
    monthly = df.groupby(['from', 'year_month']).agg(
        monthly_score=('sentiment_score', 'sum'),
        positive_count=('sentiment_score', lambda x: (x == 1).sum()),
        negative_count=('sentiment_score', lambda x: (x == -1).sum()),
        neutral_count=('sentiment_score', lambda x: (x == 0).sum()),
        total_messages=('sentiment_score', 'count')
    ).reset_index()
    
    return monthly


# ============================================================================
# TASK 4: EMPLOYEE RANKING
# ============================================================================

def rank_employees(monthly_scores):
    """
    Generate ranked lists of employees based on monthly sentiment scores.
    
    Creates two lists per month:
    1. Top 3 Positive: Highest monthly_score, ties broken alphabetically.
    2. Top 3 Negative: Lowest monthly_score, ties broken alphabetically.
    
    Parameters:
        monthly_scores (pd.DataFrame): Monthly scores DataFrame.
    
    Returns:
        tuple: (top_positive_df, top_negative_df)
    """
    top_positive_list = []
    top_negative_list = []
    
    for month in monthly_scores['year_month'].unique():
        month_data = monthly_scores[monthly_scores['year_month'] == month].copy()
        
        # Sort descending by score, then alphabetically by name
        month_data_sorted = month_data.sort_values(
            by=['monthly_score', 'from'], 
            ascending=[False, True]
        )
        top_3_pos = month_data_sorted.head(3)
        top_3_pos = top_3_pos.copy()
        top_3_pos['rank'] = range(1, len(top_3_pos) + 1)
        top_positive_list.append(top_3_pos)
        
        # Sort ascending by score (most negative first), then alphabetically
        month_data_sorted_neg = month_data.sort_values(
            by=['monthly_score', 'from'], 
            ascending=[True, True]
        )
        top_3_neg = month_data_sorted_neg.head(3)
        top_3_neg = top_3_neg.copy()
        top_3_neg['rank'] = range(1, len(top_3_neg) + 1)
        top_negative_list.append(top_3_neg)
    
    top_positive_df = pd.concat(top_positive_list, ignore_index=True)
    top_negative_df = pd.concat(top_negative_list, ignore_index=True)
    
    return top_positive_df, top_negative_df


# ============================================================================
# TASK 5: FLIGHT RISK IDENTIFICATION
# ============================================================================

def identify_flight_risks(df):
    """
    Identify employees at risk of leaving based on negative message frequency.
    
    A flight risk is any employee who has sent 4 or more negative messages
    within a rolling 30-day window (irrespective of calendar month boundaries).
    
    Parameters:
        df (pd.DataFrame): DataFrame with 'from', 'date', and 'sentiment' columns.
    
    Returns:
        pd.DataFrame: DataFrame of flight risk employees with details.
    """
    # Filter only negative messages
    negative_msgs = df[df['sentiment'] == 'Negative'].copy()
    negative_msgs = negative_msgs.sort_values(['from', 'date'])
    
    flight_risk_employees = []
    
    for employee in negative_msgs['from'].unique():
        emp_negatives = negative_msgs[negative_msgs['from'] == employee].copy()
        emp_negatives = emp_negatives.sort_values('date').reset_index(drop=True)
        
        if len(emp_negatives) < 4:
            continue
        
        # Rolling 30-day window check
        dates = emp_negatives['date'].values
        is_flight_risk = False
        risk_window_start = None
        risk_window_end = None
        risk_count = 0
        
        for i in range(len(dates)):
            # Count messages within 30 days from dates[i]
            window_start = dates[i]
            window_end = dates[i] + np.timedelta64(30, 'D')
            count = ((dates >= window_start) & (dates <= window_end)).sum()
            
            if count >= 4:
                is_flight_risk = True
                risk_window_start = pd.Timestamp(window_start)
                risk_window_end = pd.Timestamp(window_end)
                risk_count = count
                break
        
        if is_flight_risk:
            flight_risk_employees.append({
                'employee': employee,
                'total_negative_messages': len(emp_negatives),
                'max_negatives_in_30_days': risk_count,
                'risk_window_start': risk_window_start,
                'risk_window_end': risk_window_end
            })
    
    return pd.DataFrame(flight_risk_employees)


# ============================================================================
# TASK 6: FEATURE ENGINEERING FOR PREDICTIVE MODELING
# ============================================================================

def engineer_features(df):
    """
    Engineer features for the linear regression model.
    
    Features created:
    - message_length: Length of the full text (subject + body)
    - word_count: Number of words in the message
    - subject_length: Length of the subject line
    - has_subject: Whether the message has a subject (binary)
    - day_of_week: Day of the week (0=Monday, 6=Sunday)
    - hour_proxy: Not available, but we use day_of_week as temporal feature
    - is_weekend: Whether the message was sent on a weekend
    - exclamation_count: Number of exclamation marks
    - question_count: Number of question marks
    - caps_ratio: Ratio of uppercase characters to total characters
    
    Parameters:
        df (pd.DataFrame): DataFrame with text and date columns.
    
    Returns:
        pd.DataFrame: DataFrame with additional feature columns.
    """
    df['message_length'] = df['full_text'].str.len()
    df['word_count'] = df['full_text'].str.split().str.len()
    df['subject_length'] = df['Subject'].fillna('').str.len()
    df['has_subject'] = (df['Subject'].fillna('') != '').astype(int)
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month'] = df['date'].dt.month
    df['exclamation_count'] = df['full_text'].str.count('!')
    df['question_count'] = df['full_text'].str.count(r'\?')
    
    # Caps ratio
    def caps_ratio(text):
        if pd.isna(text) or len(text) == 0:
            return 0.0
        alpha_chars = [c for c in text if c.isalpha()]
        if len(alpha_chars) == 0:
            return 0.0
        return sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
    
    df['caps_ratio'] = df['full_text'].apply(caps_ratio)
    
    return df


def prepare_regression_data(df):
    """
    Prepare data for linear regression model.
    
    Uses engineered features as independent variables (X) and
    polarity score as the dependent variable (y).
    
    Parameters:
        df (pd.DataFrame): DataFrame with features and polarity.
    
    Returns:
        tuple: (X, y, feature_names)
    """
    feature_cols = [
        'message_length', 'word_count', 'subject_length',
        'has_subject', 'day_of_week', 'is_weekend', 'month',
        'exclamation_count', 'question_count', 'caps_ratio'
    ]
    
    X = df[feature_cols].fillna(0)
    y = df['polarity']
    
    return X, y, feature_cols
