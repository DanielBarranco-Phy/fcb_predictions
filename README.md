# ⚽ FCB Predictions

Welcome to **FCB Predictions**! This project uses **machine learning, Elo ratings and four seasons of match data** to predict the result of every FC Barcelona match: La Liga, Champions League, Copa del Rey and Supercopa.

Before each match I post the prediction on Instagram and TikTok. After the match, the model's track record is updated here, hit or miss.

## 🔮 What it predicts

For Barça's next match, three probabilities: **win, draw, loss** (after 90 minutes).

```
=== Barcelona vs Getafe · liga · away · 10/10/2026 ===

Form (last 5): W W W W W
Head-to-head vs Getafe since 22/23: W D D W W D W W
Elo   Barcelona 1866   Getafe 1489   (difference +377)

MODEL PREDICTION
  Barça win    68.8%
  Draw         23.4%
  Barça loss    7.7%
```

## 📊 Data sources

Every match since the 2022/23 season, about 15,000 in total:

| Source | What it provides |
|---|---|
| [football-data.co.uk](https://www.football-data.co.uk) | 11 European leagues: results and bookmaker odds |
| [fixturedownload.com](https://fixturedownload.com) | Champions League results and upcoming fixtures |
| ESPN public API | Barça's Copa del Rey and Supercopa matches |

Only La Liga and Champions League matches train the model. The other ten leagues exist to answer one question: how strong is a European opponent Barça has never played?

## 🧠 How it works

1. **Collect** (`src/data.py`): download everything into one table, one row per match.
2. **Rate** (`src/elo.py`): every team gets an Elo rating, updated after each match. Beating a stronger team moves it up a lot; beating a weaker one barely moves it. League starting levels are fitted from Champions League results, so a Turkish team and a Spanish team are on the same scale.
3. **Build features** (`src/features.py`): Elo difference, form over the last 5 matches (points and goal difference), rest days, competition. Only information available before kickoff.
4. **Train** (`src/model.py`): multinomial logistic regression (scikit-learn) on about 2,000 La Liga and Champions League matches. Recent matches count more: a match from two years ago weighs half as much as one from today.
5. **Predict** (`predict.py`): home, away or neutral venue. For neutral venues the match is simulated both ways and averaged.

## 📈 Model performance

Trained on matches up to June 2025, tested on 655 later matches it had never seen:

| | Accuracy |
|---|---|
| Always pick the home team | 49.2% |
| **This model** (La Liga + Champions League) | **54.4%** |
| This model, La Liga only | 51.2% |
| Bookmakers, La Liga only | 54.6% |

Football is noisy and nobody gets far above 55%. The bookmakers still beat this model in La Liga; closing that gap in public is the point of the project.

Run `python src/model.py` to reproduce these numbers.

## 🚀 Usage

```bash
pip install -r requirements.txt
python update.py     # download the data and show the model's track record
python predict.py    # predict a match
```

To choose the match, edit the lines at the bottom of `predict.py`:

```python
OPPONENT = "Getafe"
COMPETITION = "liga"      # liga | champions | copa | supercopa
VENUE = "away"            # home | away | neutral
DATE = "10/10/2026"
ABSENT_BARCA = {"Raphinha": "star"}    # optional: star | starter | rotation
ABSENT_OPPONENT = {}
```

Missing players lower a team's Elo by a fixed amount (star 40, starter 20, rotation 10). These values are assumptions, not learned from data, so the output always shows both predictions: the base model and the one with absences.

Every prediction is saved to `predictions.csv`. `update.py` compares each one with the real result and writes `track_record.csv`.

If a source has not published a result yet, add it as one line in `manual_results.csv`. It is dropped automatically once the official data arrives.

## 🗂️ Files

```
predict.py           predict Barça's next match
update.py            refresh data, score past predictions
src/data.py          download and load matches
src/elo.py           Elo ratings
src/features.py      model inputs
src/model.py         training and evaluation
predictions.csv      every prediction made, with its date
track_record.csv     predictions vs real results
COMO_FUNCIONA.md     detailed explanation of the method (in Spanish)
```

## 🛠️ What's next

The model starts simple on purpose and will grow during the season:

- Expected goals (xG) instead of goals only
- Starting elevens and player ratings as a real model feature (today absences are a manual adjustment)
- Squad rotation before big matches
- Predicted scorelines

## ⚠️ Disclaimer

This is an educational data science project. It is not betting advice.
