"""
visualizations_generator.py
============================
Generates all visualization charts for the Employee Sentiment Analysis project.
Saves plots to the 'visualizations/' folder.

Author: Employee Sentiment Analysis Project
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for all plots
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11

VIS_DIR = 'visualizations'


def save_fig(fig, filename):
    """Save figure to the visualizations directory."""
    filepath = os.path.join(VIS_DIR, filename)
    fig.savefig(filepath, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  [SAVED] {filepath}")
    return filepath


# ============================================================================
# EDA VISUALIZATIONS
# ============================================================================

def plot_sentiment_distribution(df):
    """Bar chart of sentiment label distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count plot
    counts = df['sentiment'].value_counts()
    colors = {'Positive': '#2ecc71', 'Neutral': '#3498db', 'Negative': '#e74c3c'}
    bars = axes[0].bar(counts.index, counts.values, 
                       color=[colors.get(x, '#95a5a6') for x in counts.index],
                       edgecolor='white', linewidth=1.5)
    axes[0].set_title('Sentiment Distribution (Count)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Sentiment')
    axes[0].set_ylabel('Number of Messages')
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                     str(val), ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    axes[1].pie(counts.values, labels=counts.index, autopct='%1.1f%%',
                colors=[colors.get(x, '#95a5a6') for x in counts.index],
                startangle=140, shadow=True, explode=[0.05]*len(counts))
    axes[1].set_title('Sentiment Distribution (%)', fontsize=14, fontweight='bold')
    
    fig.suptitle('Overall Sentiment Distribution of Employee Messages',
                 fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    return save_fig(fig, '01_sentiment_distribution.png')


def plot_sentiment_over_time(df):
    """Line chart showing sentiment trends over time."""
    df_copy = df.copy()
    df_copy['year_month'] = df_copy['date'].dt.to_period('M')
    
    monthly_sentiment = df_copy.groupby(['year_month', 'sentiment']).size().unstack(fill_value=0)
    monthly_sentiment.index = monthly_sentiment.index.astype(str)
    
    fig, ax = plt.subplots(figsize=(16, 6))
    colors = {'Positive': '#2ecc71', 'Neutral': '#3498db', 'Negative': '#e74c3c'}
    
    for col in ['Positive', 'Neutral', 'Negative']:
        if col in monthly_sentiment.columns:
            ax.plot(monthly_sentiment.index, monthly_sentiment[col], 
                    label=col, color=colors[col], marker='o', linewidth=2, markersize=4)
    
    ax.set_title('Sentiment Trends Over Time (Monthly)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Number of Messages', fontsize=12)
    ax.legend(fontsize=11, loc='upper right')
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    return save_fig(fig, '02_sentiment_over_time.png')


def plot_polarity_distribution(df):
    """Histogram of polarity scores."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(df['polarity'], bins=50, color='#3498db', edgecolor='white', alpha=0.8)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Neutral (0)')
    ax.axvline(x=df['polarity'].mean(), color='green', linestyle='--', linewidth=1.5, 
               label=f'Mean ({df["polarity"].mean():.3f})')
    ax.set_title('Distribution of Polarity Scores', fontsize=16, fontweight='bold')
    ax.set_xlabel('Polarity Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend(fontsize=11)
    fig.tight_layout()
    return save_fig(fig, '03_polarity_distribution.png')


def plot_messages_per_employee(df):
    """Bar chart of message count per employee (top senders)."""
    msg_counts = df['from'].value_counts().head(15)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.barh(range(len(msg_counts)), msg_counts.values, color='#3498db',
                   edgecolor='white', linewidth=1)
    ax.set_yticks(range(len(msg_counts)))
    ax.set_yticklabels([e.split('@')[0] for e in msg_counts.index], fontsize=10)
    ax.set_xlabel('Number of Messages', fontsize=12)
    ax.set_title('Top 15 Employees by Message Count', fontsize=16, fontweight='bold')
    ax.invert_yaxis()
    
    for i, val in enumerate(msg_counts.values):
        ax.text(val + 5, i, str(val), va='center', fontweight='bold')
    
    fig.tight_layout()
    return save_fig(fig, '04_messages_per_employee.png')


def plot_sentiment_by_employee(df):
    """Stacked bar chart of sentiment breakdown per employee (top senders)."""
    top_employees = df['from'].value_counts().head(10).index
    emp_sentiment = df[df['from'].isin(top_employees)].groupby(['from', 'sentiment']).size().unstack(fill_value=0)
    emp_sentiment = emp_sentiment.loc[top_employees]
    
    # Shorten names
    emp_sentiment.index = [e.split('@')[0] for e in emp_sentiment.index]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = {'Positive': '#2ecc71', 'Neutral': '#3498db', 'Negative': '#e74c3c'}
    
    bottom = np.zeros(len(emp_sentiment))
    for sentiment in ['Positive', 'Neutral', 'Negative']:
        if sentiment in emp_sentiment.columns:
            vals = emp_sentiment[sentiment].values
            ax.bar(emp_sentiment.index, vals, bottom=bottom,
                   label=sentiment, color=colors[sentiment], edgecolor='white')
            bottom += vals
    
    ax.set_title('Sentiment Breakdown by Top 10 Employees', fontsize=16, fontweight='bold')
    ax.set_xlabel('Employee', fontsize=12)
    ax.set_ylabel('Number of Messages', fontsize=12)
    ax.legend(fontsize=11)
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    return save_fig(fig, '05_sentiment_by_employee.png')


def plot_day_of_week(df):
    """Bar chart of message count by day of week."""
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df_copy = df.copy()
    df_copy['day_of_week'] = df_copy['date'].dt.dayofweek
    day_counts = df_copy['day_of_week'].value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar([day_names[i] for i in day_counts.index], day_counts.values,
                  color=['#3498db'] * 5 + ['#e74c3c'] * 2, edgecolor='white', linewidth=1.5)
    ax.set_title('Messages by Day of Week', fontsize=16, fontweight='bold')
    ax.set_xlabel('Day of Week', fontsize=12)
    ax.set_ylabel('Number of Messages', fontsize=12)
    
    for bar, val in zip(bars, day_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(val), ha='center', va='bottom', fontweight='bold')
    
    fig.tight_layout()
    return save_fig(fig, '06_day_of_week.png')


def plot_message_length_by_sentiment(df):
    """Box plot of message length by sentiment."""
    fig, ax = plt.subplots(figsize=(10, 6))
    df_copy = df.copy()
    df_copy['message_length'] = df_copy['full_text'].str.len()
    
    # Clip outliers for better visualization
    q99 = df_copy['message_length'].quantile(0.99)
    df_plot = df_copy[df_copy['message_length'] <= q99]
    
    colors = {'Positive': '#2ecc71', 'Neutral': '#3498db', 'Negative': '#e74c3c'}
    palette = [colors.get(s, '#95a5a6') for s in ['Negative', 'Neutral', 'Positive']]
    
    sns.boxplot(data=df_plot, x='sentiment', y='message_length',
                order=['Negative', 'Neutral', 'Positive'],
                palette=palette, ax=ax)
    ax.set_title('Message Length by Sentiment', fontsize=16, fontweight='bold')
    ax.set_xlabel('Sentiment', fontsize=12)
    ax.set_ylabel('Message Length (characters)', fontsize=12)
    fig.tight_layout()
    return save_fig(fig, '07_message_length_by_sentiment.png')


def plot_wordcloud_by_sentiment(df):
    """Generate word clouds for each sentiment category."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    sentiments = ['Positive', 'Neutral', 'Negative']
    color_maps = ['Greens', 'Blues', 'Reds']
    
    for ax, sentiment, cmap in zip(axes, sentiments, color_maps):
        text = ' '.join(df[df['sentiment'] == sentiment]['full_text'].dropna().astype(str))
        if text.strip():
            wc = WordCloud(width=600, height=400, background_color='white',
                          colormap=cmap, max_words=100, 
                          stopwords=set(['enron', 'com', 'will', 'please', 'us', 'one', 'would']))
            wc.generate(text)
            ax.imshow(wc, interpolation='bilinear')
        ax.set_title(f'{sentiment} Messages', fontsize=14, fontweight='bold')
        ax.axis('off')
    
    fig.suptitle('Word Clouds by Sentiment Category', fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    return save_fig(fig, '08_wordclouds.png')


# ============================================================================
# RANKING VISUALIZATIONS
# ============================================================================

def plot_monthly_scores_heatmap(monthly_scores):
    """Heatmap of monthly scores by employee."""
    # Get top employees by absolute activity
    top_emps = monthly_scores.groupby('from')['total_messages'].sum().nlargest(15).index
    filtered = monthly_scores[monthly_scores['from'].isin(top_emps)].copy()
    
    pivot = filtered.pivot_table(index='from', columns='year_month', values='monthly_score', fill_value=0)
    pivot.index = [e.split('@')[0] for e in pivot.index]
    pivot.columns = pivot.columns.astype(str)
    
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(pivot, cmap='RdYlGn', center=0, annot=True, fmt='.0f',
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Monthly Score'})
    ax.set_title('Monthly Sentiment Scores Heatmap (Top 15 Employees)',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Employee', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    return save_fig(fig, '09_monthly_scores_heatmap.png')


def plot_top_employees_ranking(top_positive, top_negative):
    """Bar charts showing top positive and negative employees."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Aggregate across all months for overall ranking
    pos_agg = top_positive.groupby('from')['monthly_score'].mean().nlargest(5)
    neg_agg = top_negative.groupby('from')['monthly_score'].mean().nsmallest(5)
    
    # Top Positive
    labels_pos = [e.split('@')[0] for e in pos_agg.index]
    axes[0].barh(range(len(pos_agg)), pos_agg.values, color='#2ecc71', edgecolor='white')
    axes[0].set_yticks(range(len(pos_agg)))
    axes[0].set_yticklabels(labels_pos, fontsize=10)
    axes[0].set_title('Top Positive Employees (Avg Monthly Score)',
                      fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Average Monthly Score')
    axes[0].invert_yaxis()
    
    # Top Negative
    labels_neg = [e.split('@')[0] for e in neg_agg.index]
    axes[1].barh(range(len(neg_agg)), neg_agg.values, color='#e74c3c', edgecolor='white')
    axes[1].set_yticks(range(len(neg_agg)))
    axes[1].set_yticklabels(labels_neg, fontsize=10)
    axes[1].set_title('Top Negative Employees (Avg Monthly Score)',
                      fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Average Monthly Score')
    axes[1].invert_yaxis()
    
    fig.suptitle('Employee Rankings by Sentiment Score', fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    return save_fig(fig, '10_employee_rankings.png')


def plot_flight_risk(flight_risks):
    """Bar chart of flight risk employees."""
    if flight_risks.empty:
        print("  [WARNING] No flight risk employees identified.")
        return None
    
    fr = flight_risks.sort_values('max_negatives_in_30_days', ascending=False)
    
    fig, ax = plt.subplots(figsize=(14, max(5, len(fr) * 0.5)))
    labels = [e.split('@')[0] for e in fr['employee']]
    colors_list = ['#e74c3c' if x >= 6 else '#e67e22' if x >= 5 else '#f1c40f' 
                   for x in fr['max_negatives_in_30_days']]
    
    bars = ax.barh(range(len(fr)), fr['max_negatives_in_30_days'].values,
                   color=colors_list, edgecolor='white', linewidth=1)
    ax.set_yticks(range(len(fr)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Max Negative Messages in 30-Day Window', fontsize=12)
    ax.set_title('Flight Risk Employees (≥4 Negative Messages in 30 Days)',
                 fontsize=16, fontweight='bold')
    ax.axvline(x=4, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Threshold (4)')
    ax.legend(fontsize=11)
    ax.invert_yaxis()
    
    for i, val in enumerate(fr['max_negatives_in_30_days'].values):
        ax.text(val + 0.1, i, str(val), va='center', fontweight='bold')
    
    fig.tight_layout()
    return save_fig(fig, '11_flight_risk_employees.png')


# ============================================================================
# REGRESSION VISUALIZATIONS
# ============================================================================

def plot_regression_results(y_test, y_pred, feature_importance, feature_names):
    """Plot regression results: actual vs predicted and feature importance."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Actual vs Predicted
    axes[0].scatter(y_test, y_pred, alpha=0.3, color='#3498db', s=10)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Polarity', fontsize=12)
    axes[0].set_ylabel('Predicted Polarity', fontsize=12)
    axes[0].set_title('Actual vs Predicted Polarity Scores', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    
    # Feature Importance
    importance = pd.Series(feature_importance, index=feature_names).sort_values()
    colors_fi = ['#e74c3c' if v < 0 else '#2ecc71' for v in importance.values]
    axes[1].barh(range(len(importance)), importance.values, color=colors_fi, edgecolor='white')
    axes[1].set_yticks(range(len(importance)))
    axes[1].set_yticklabels(importance.index, fontsize=10)
    axes[1].set_xlabel('Coefficient Value', fontsize=12)
    axes[1].set_title('Feature Importance (Linear Regression Coefficients)',
                      fontsize=14, fontweight='bold')
    axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    fig.suptitle('Linear Regression Model Results', fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    return save_fig(fig, '12_regression_results.png')


def plot_residuals(y_test, y_pred):
    """Plot residual distribution."""
    residuals = y_test - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residual histogram
    axes[0].hist(residuals, bins=50, color='#3498db', edgecolor='white', alpha=0.8)
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=1.5)
    axes[0].set_title('Residual Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Residual')
    axes[0].set_ylabel('Frequency')
    
    # Residual scatter
    axes[1].scatter(y_pred, residuals, alpha=0.3, color='#e74c3c', s=10)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_title('Residuals vs Predicted Values', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted Polarity')
    axes[1].set_ylabel('Residual')
    
    fig.suptitle('Model Diagnostic Plots', fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    return save_fig(fig, '13_residuals.png')


def generate_all_eda_visualizations(df):
    """Generate all EDA visualizations."""
    print("\n[CHARTS] Generating EDA Visualizations...")
    print("-" * 50)
    
    plot_sentiment_distribution(df)
    plot_sentiment_over_time(df)
    plot_polarity_distribution(df)
    plot_messages_per_employee(df)
    plot_sentiment_by_employee(df)
    plot_day_of_week(df)
    plot_message_length_by_sentiment(df)
    plot_wordcloud_by_sentiment(df)
    
    print("\n[OK] All EDA visualizations generated!")
