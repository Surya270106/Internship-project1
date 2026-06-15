# Employee Sentiment Analysis

## Project Overview

This project works with an unlabeled dataset of employee emails (Enron corpus) to figure out the overall sentiment and engagement levels. I used Python with TextBlob for the NLP side and scikit-learn for the regression model.

The dataset (`test.csv`) has around 2,191 messages with columns: Subject, body, date, and from (sender email).

## What This Project Does

1. **Sentiment Labeling** - Labels each message as Positive, Negative, or Neutral using TextBlob
2. **EDA** - Explores the data, checks distributions, trends over time, etc.
3. **Monthly Scoring** - Calculates a sentiment score per employee per month (+1 for positive, -1 for negative, 0 for neutral)
4. **Employee Ranking** - Picks out the top 3 positive and top 3 negative employees each month
5. **Flight Risk** - Flags anyone who sent 4+ negative emails in a rolling 30-day window
6. **Linear Regression** - Builds a model using message metadata to see what features correlate with sentiment

---

## How to Set Up

**You need Python 3.8 or higher.**

1. Clone the repo:
   ```bash
   git clone https://github.com/Surya270106/Internship-project1.git
   cd Internship-project1
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the NLTK data that TextBlob needs:
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('brown'); nltk.download('movie_reviews'); nltk.download('wordnet')"
   ```

4. Make sure `test.csv` is in the root folder (it should already be there).

---

## How to Run

**Option 1 - Run the Python script:**
```bash
python main_analysis.py
```
This runs everything end-to-end and saves all the output files + visualizations.

**Option 2 - Use the Jupyter Notebook:**
```bash
jupyter notebook Employee_Sentiment_Analysis.ipynb
```
Same analysis but with inline charts and step-by-step commentary.

After running, you'll get:
- `test_labeled.csv` - the dataset with sentiment labels added
- `monthly_scores.csv` - monthly scores per employee
- `top_positive_employees.csv` / `top_negative_employees.csv` - rankings
- `flight_risk_employees.csv` - flagged employees
- `visualizations/` folder with 13 charts

---

## Project Structure

```
Internship-project1/
|-- test.csv                          # raw dataset
|-- main_analysis.py                  # runs the full pipeline
|-- sentiment_analysis.py             # helper functions (labeling, scoring, etc.)
|-- visualizations_generator.py       # all the plotting code
|-- Employee_Sentiment_Analysis.ipynb  # notebook version (main deliverable)
|-- requirements.txt
|-- README.md
|-- .env.example
|-- visualizations/                   # output charts (13 PNGs)
|-- test_labeled.csv                  # generated output
|-- monthly_scores.csv                # generated output
|-- top_positive_employees.csv        # generated output
|-- top_negative_employees.csv        # generated output
|-- flight_risk_employees.csv         # generated output
```

---

## Methodology

### Sentiment Labeling
I used TextBlob which gives a polarity score from -1 to +1 for any text. I combined the Subject and body fields for better context, then classified based on thresholds:
- Polarity > 0.1 = Positive
- Polarity < -0.1 = Negative
- Everything else = Neutral

I picked 0.1 as the cutoff (instead of 0) to avoid labeling very mildly worded emails as positive/negative when they're basically neutral.

### EDA
Looked at the overall shape of the data, checked for missing values, plotted sentiment distributions, tracked trends over time (monthly), broke down activity by day of week, compared message lengths across sentiment types, and generated word clouds to see what kind of language appears in each category.

### Monthly Scoring
Pretty straightforward - each positive message gets +1, negative gets -1, neutral gets 0. These are summed up per employee per month. The score resets every month.

### Employee Ranking
For each month, I sorted employees by their monthly score. Top 3 highest = most positive, top 3 lowest = most negative. When there's a tie, I sorted alphabetically.

### Flight Risk
The requirement says anyone with 4 or more negative messages in a 30-day span is a flight risk. I implemented this as a rolling window - for each employee, I go through their negative message dates and check if any 30-day window starting from each date contains 4+ negatives. This is independent of calendar months.

### Linear Regression
I engineered 10 features from the data (message length, word count, subject length, whether it has a subject, day of week, weekend flag, month, exclamation count, question mark count, uppercase ratio) and used them to predict the polarity score. Used scikit-learn's LinearRegression with StandardScaler, 80/20 train-test split.

---

## Results

### Sentiment Breakdown
| Sentiment | Count | % |
|-----------|-------|---|
| Neutral | 1,019 | 46.5% |
| Positive | 976 | 44.5% |
| Negative | 196 | 8.9% |

Most emails are neutral or positive, which makes sense for workplace communication. Only about 9% came out as negative.

### Top 3 Positive Employees (Overall)
| # | Employee | Total Score |
|---|----------|-------------|
| 1 | lydia.delgado@enron.com | 101 |
| 2 | john.arnold@enron.com | 97 |
| 3 | sally.beck@enron.com | 89 |

### Top 3 Negative Employees (Overall)
| # | Employee | Total Score |
|---|----------|-------------|
| 1 | rhonda.denton@enron.com | 50 |
| 2 | kayne.coulter@enron.com | 61 |
| 3 | bobette.riner@ipgdirect.com | 72 |

### Flight Risk Employees
6 employees got flagged:

| Employee | Total Negatives | Max in 30 Days | Window |
|----------|----------------|----------------|--------|
| bobette.riner@ipgdirect.com | 21 | 6 | 2010-11-02 to 2010-12-02 |
| johnny.palmer@enron.com | 18 | 4 | 2011-01-29 to 2011-02-28 |
| lydia.delgado@enron.com | 28 | 6 | 2011-12-04 to 2012-01-03 |
| patti.thompson@enron.com | 23 | 4 | 2011-03-17 to 2011-04-16 |
| rhonda.denton@enron.com | 20 | 4 | 2010-12-18 to 2011-01-17 |
| sally.beck@enron.com | 18 | 4 | 2010-06-16 to 2010-07-16 |

### Regression Model
| Metric | Train | Test |
|--------|-------|------|
| R2 | 0.0489 | 0.0742 |
| RMSE | 0.2132 | 0.2285 |
| MAE | 0.1523 | 0.1619 |

The R2 is low, but that's expected - you can't really predict what someone *said* just from how long the email was or what day they sent it. The model does show that exclamation marks and word count have the strongest correlation with polarity, which is interesting.

---

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- textblob
- scikit-learn
- wordcloud
- nltk
- openpyxl

Full versions are in `requirements.txt`.

---

## Notes

- TextBlob runs entirely locally, no API keys needed.
- All charts are saved as PNGs in `visualizations/`.
- Random state is 42 for the train/test split so results are reproducible.
- See `.env.example` for optional config (thresholds, file paths, etc.).
