# Employee Sentiment Analysis

## 📋 Project Overview

This project analyzes an unlabeled dataset of employee email messages (from the Enron corpus) to assess **sentiment and engagement**. The analysis pipeline works from raw data to derive actionable insights using **Natural Language Processing (NLP)** and **statistical analysis** techniques.

## 🎯 Project Objectives

| Task | Description |
|------|-------------|
| **Task 1** | **Sentiment Labeling** — Automatically classify each message as Positive, Negative, or Neutral |
| **Task 2** | **Exploratory Data Analysis (EDA)** — Analyze and visualize data structure, distributions, and trends |
| **Task 3** | **Monthly Sentiment Scoring** — Compute monthly sentiment scores per employee (+1, -1, 0) |
| **Task 4** | **Employee Ranking** — Rank top 3 positive and negative employees per month |
| **Task 5** | **Flight Risk Identification** — Flag employees with ≥4 negative messages in any rolling 30-day window |
| **Task 6** | **Predictive Modeling** — Build a linear regression model to analyze sentiment trends |

---

## 🛠️ Setup Instructions

### Prerequisites

- **Python 3.8+** (tested with Python 3.14)
- **pip** (Python package manager)

### Installation

1. **Clone or extract the repository:**
   ```bash
   git clone <repository-url>
   cd employee-sentiment-analysis
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data (required for TextBlob):**
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('brown'); nltk.download('movie_reviews'); nltk.download('wordnet')"
   ```

4. **Ensure the dataset file `test.csv` is in the project root directory.**

---

## 🚀 Usage

### Run the Full Analysis Pipeline

```bash
python main_analysis.py
```

This single command executes all 6 tasks and generates:
- Labeled dataset (`test_labeled.csv`)
- Monthly scores (`monthly_scores.csv`)
- Employee rankings (`top_positive_employees.csv`, `top_negative_employees.csv`)
- Flight risk list (`flight_risk_employees.csv`)
- All visualizations in the `visualizations/` folder

### Alternative: Run as Jupyter Notebook

```bash
jupyter notebook Employee_Sentiment_Analysis.ipynb
```

The notebook contains the same analysis with inline visualizations, commentary, and observations.

---

## 📂 Project Structure

```
employee-sentiment-analysis/
│
├── test.csv                          # Original unlabeled dataset
├── main_analysis.py                  # Main analysis script (runs all tasks)
├── sentiment_analysis.py             # Core NLP & analysis functions
├── visualizations_generator.py       # Visualization generation utilities
├── Employee_Sentiment_Analysis.ipynb # Jupyter Notebook (main deliverable)
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── .env.example                      # Environment variable template
│
├── visualizations/                   # Generated charts and plots
│   ├── 01_sentiment_distribution.png
│   ├── 02_sentiment_over_time.png
│   ├── 03_polarity_distribution.png
│   ├── 04_messages_per_employee.png
│   ├── 05_sentiment_by_employee.png
│   ├── 06_day_of_week.png
│   ├── 07_message_length_by_sentiment.png
│   ├── 08_wordclouds.png
│   ├── 09_monthly_scores_heatmap.png
│   ├── 10_employee_rankings.png
│   ├── 11_flight_risk_employees.png
│   ├── 12_regression_results.png
│   └── 13_residuals.png
│
└── Output CSV Files (generated):
    ├── test_labeled.csv              # Dataset with sentiment labels
    ├── monthly_scores.csv            # Monthly sentiment scores
    ├── top_positive_employees.csv    # Top positive rankings
    ├── top_negative_employees.csv    # Top negative rankings
    └── flight_risk_employees.csv     # Flight risk employees
```

---

## 🔬 Methodology

### Task 1: Sentiment Labeling
- **Library:** TextBlob (built on NLTK)
- **Approach:** Combined Subject + Body text → TextBlob polarity analysis
- **Thresholds:**
  - Polarity > 0.1 → **Positive**
  - Polarity < -0.1 → **Negative**
  - Otherwise → **Neutral**

### Task 2: EDA
- Examined dataset structure, missing values, and data types
- Analyzed sentiment distribution across categories
- Investigated temporal trends (monthly, day-of-week)
- Explored employee activity patterns
- Generated word clouds and statistical visualizations

### Task 3: Monthly Scoring
- **Positive message:** +1
- **Negative message:** -1
- **Neutral message:** 0
- Scores aggregated monthly per employee, resetting each month

### Task 4: Employee Ranking
- Sorted by monthly score (descending for positive, ascending for negative)
- Ties broken alphabetically
- Top 3 per category per month

### Task 5: Flight Risk
- **Criteria:** ≥4 negative messages within any rolling 30-day window
- Uses a sliding window approach across all dates
- Independent of calendar month boundaries

### Task 6: Linear Regression
- **Features:** message_length, word_count, subject_length, has_subject, day_of_week, is_weekend, month, exclamation_count, question_count, caps_ratio
- **Target:** Polarity score (continuous)
- **Library:** scikit-learn (LinearRegression with StandardScaler)
- **Split:** 80% train / 20% test
- **Metrics:** R², RMSE, MAE

---

## 📊 Key Findings Summary

### Sentiment Distribution
| Sentiment | Count | Percentage |
|-----------|-------|------------|
| Neutral   | 1,019 | 46.5%     |
| Positive  | 976   | 44.5%     |
| Negative  | 196   | 8.9%      |

### Top 3 Positive Employees (Overall)
| Rank | Employee | Total Score |
|------|----------|-------------|
| #1   | lydia.delgado@enron.com | 101 |
| #2   | john.arnold@enron.com | 97 |
| #3   | sally.beck@enron.com | 89 |

### Top 3 Negative Employees (Overall)
| Rank | Employee | Total Score |
|------|----------|-------------|
| #1   | rhonda.denton@enron.com | 50 |
| #2   | kayne.coulter@enron.com | 61 |
| #3   | bobette.riner@ipgdirect.com | 72 |

### Flight Risk Employees
The following employees were flagged as flight risks (≥4 negative messages in a rolling 30-day window):

| Employee | Total Negatives | Max in 30-Day Window | Risk Window |
|----------|----------------|---------------------|-------------|
| bobette.riner@ipgdirect.com | 21 | 6 | 2010-11-02 to 2010-12-02 |
| johnny.palmer@enron.com | 18 | 4 | 2011-01-29 to 2011-02-28 |
| lydia.delgado@enron.com | 28 | 6 | 2011-12-04 to 2012-01-03 |
| patti.thompson@enron.com | 23 | 4 | 2011-03-17 to 2011-04-16 |
| rhonda.denton@enron.com | 20 | 4 | 2010-12-18 to 2011-01-17 |
| sally.beck@enron.com | 18 | 4 | 2010-06-16 to 2010-07-16 |

### Model Performance (Linear Regression)
| Metric | Training | Testing |
|--------|----------|---------|
| R² Score | 0.0489 | 0.0742 |
| RMSE | 0.2132 | 0.2285 |
| MAE | 0.1523 | 0.1619 |

**Note:** The low R² is expected since sentiment is primarily determined by text content, not structural metadata features. The model serves as a baseline showing which structural attributes (exclamation count, word count, caps ratio) correlate with sentiment.

### Key Recommendations
- Monitor flight risk employees closely for signs of disengagement
- Investigate root causes of negative sentiment patterns
- Consider implementing regular sentiment check-ins
- Use sentiment trends to inform retention strategies

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | ≥2.0 | Data manipulation and analysis |
| numpy | ≥1.24 | Numerical computing |
| matplotlib | ≥3.7 | Data visualization |
| seaborn | ≥0.12 | Statistical data visualization |
| textblob | ≥0.17 | NLP sentiment analysis |
| scikit-learn | ≥1.3 | Machine learning (Linear Regression) |
| wordcloud | ≥1.9 | Word cloud generation |
| nltk | ≥3.8 | Natural Language Toolkit |
| openpyxl | ≥3.1 | Excel file support |

---

## ⚙️ Environment Variables

See `.env.example` for optional configuration. No API keys are required for this project as TextBlob runs locally.

---

## 📝 Notes

- The sentiment analysis uses TextBlob which provides reliable results without requiring external API calls.
- All visualizations are saved as high-resolution PNG files in the `visualizations/` directory.
- The analysis is fully reproducible — running `main_analysis.py` from scratch will regenerate all outputs.
- Random state is set to 42 for reproducible train/test splits.

---

## 📄 License

This project is for internal evaluation purposes only.
