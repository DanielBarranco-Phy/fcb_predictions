"""Train and evaluate the W/D/L model.

Run `python src/model.py` to see how it would have done on matches it never saw.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import data
import elo
import features

HALF_LIFE_DAYS = 730         # a match this old counts half as much as one played today
BURN_IN = "2022-11-01"       # Elo starts from league averages, so the first weeks of ratings are not trustworthy
TEST_FROM = "2025-07-01"     # evaluation: train on everything before, test on 25/26 + 26/27
CLASSES = ["A", "D", "H"]    # sklearn sorts labels alphabetically


def training_rows(df):
    """The model learns from La Liga and Champions League home/away matches (all teams, not just Barça)."""
    return df[df.comp.isin(["liga", "champions"]) & (df.neutral == 0) & (df.date >= BURN_IN)]


def fit(df, today=None, half_life=HALF_LIFE_DAYS):
    today = today or df.date.max()
    age = (today - df.date).dt.days
    weights = 0.5 ** (age / half_life) if half_life else None
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(df[features.FEATURES], df.result, logisticregression__sample_weight=weights)
    return model


def train():
    """Final model: trained on every match available."""
    df, state = features.build(data.load_matches())
    return fit(training_rows(df)), df, state


def evaluate():
    matches = data.load_matches()
    # league starting levels are fitted on the training period only, so the test stays honest
    past = matches[matches.date < TEST_FROM]
    bases = elo.fit_bases(past, elo.team_groups(past))
    df, _ = features.build(matches, bases)
    rows = training_rows(df)
    tr, te = rows[rows.date < TEST_FROM], rows[rows.date >= TEST_FROM]
    print(f"train: {len(tr)} matches   test: {len(te)} matches ({te.date.min():%Y-%m-%d} to {te.date.max():%Y-%m-%d})\n")

    print("half-life      accuracy  log loss")
    for hl in [180, 365, 730, 1460, None]:
        p = fit(tr, half_life=hl).predict_proba(te[features.FEATURES])
        acc = (np.array(CLASSES)[p.argmax(1)] == te.result).mean()
        print(f"{str(hl):<14} {acc:.1%}     {log_loss(te.result, p, labels=CLASSES):.4f}")

    model = fit(tr)
    p = model.predict_proba(te[features.FEATURES])
    pred = np.array(CLASSES)[p.argmax(1)]
    freq = tr.result.value_counts(normalize=True)[CLASSES].values
    print("\nbaselines on the same test matches")
    print(f"  always home win        accuracy {(te.result == 'H').mean():.1%}")
    print(f"  historical frequencies log loss {log_loss(te.result, np.tile(freq, (len(te), 1)), labels=CLASSES):.4f}")
    print(f"  model                  accuracy {(pred == te.result).mean():.1%}   log loss {log_loss(te.result, p, labels=CLASSES):.4f}")

    liga = te[te.odds_h.notna()]
    odds = 1 / liga[["odds_a", "odds_d", "odds_h"]].values
    odds /= odds.sum(1, keepdims=True)  # remove the bookmaker margin
    pl = model.predict_proba(liga[features.FEATURES])
    print(f"\nLa Liga only ({len(liga)} matches)")
    print(f"  bookmakers  accuracy {(np.array(CLASSES)[odds.argmax(1)] == liga.result).mean():.1%}   log loss {log_loss(liga.result, odds, labels=CLASSES):.4f}")
    print(f"  model       accuracy {(np.array(CLASSES)[pl.argmax(1)] == liga.result).mean():.1%}   log loss {log_loss(liga.result, pl, labels=CLASSES):.4f}")

    barca = te[(te.home == "Barcelona") | (te.away == "Barcelona")]
    pb = np.array(CLASSES)[model.predict_proba(barca[features.FEATURES]).argmax(1)]
    print(f"\nBarça only ({len(barca)} matches): accuracy {(pb == barca.result).mean():.1%}")

    print("\nwhat the model learned (standardised weights, home-win class)")
    lr = model[-1]
    for f, c in zip(features.FEATURES, lr.coef_[CLASSES.index("H")]):
        print(f"  {f:<15} {c:+.3f}")


if __name__ == "__main__":
    evaluate()
