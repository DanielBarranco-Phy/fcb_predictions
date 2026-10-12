"""Predict Barça's next match.

Fill in the four values at the bottom of this file and run it (play button in VS Code, or `python predict.py`).
"""
import csv
import difflib
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
import data
import features
import model as mdl

BARCA = "Barcelona"
LOG = Path(__file__).parent / "predictions.csv"

# Elo points a team loses when a player is missing. These are my own assumptions, NOT learned from
# data: the model has never seen a lineup. For scale, playing at home is worth 60 points, so losing
# a star costs about two thirds of home advantage.
ABSENCE_PENALTY = {"star": 40, "starter": 20, "rotation": 10}


def probabilities(model, state, home, away, comp, when):
    """[P(away win), P(draw), P(home win)] for one home/away ordering."""
    r, h = state["ratings"], state["history"]
    x = features.row(when, comp, r[home], r[away], h[home], h[away])
    return model.predict_proba(pd.DataFrame([x])[features.FEATURES])[0], x


def penalty(absent):
    bad = [t for t in absent.values() if t not in ABSENCE_PENALTY]
    if bad:
        sys.exit(f"Absence type must be star, starter or rotation (you wrote '{bad[0]}')")
    return sum(ABSENCE_PENALTY[t] for t in absent.values())


def predict(rival, comp, venue, when, absent_barca=None, absent_rival=None):
    absent_barca, absent_rival = absent_barca or {}, absent_rival or {}
    model, df, state = mdl.train()
    if rival not in state["ratings"]:
        close = difflib.get_close_matches(rival, state["ratings"], n=5, cutoff=0.4)
        sys.exit(f"I don't know '{rival}'. Did you mean: {', '.join(close) or 'nothing similar found'}?")

    def wdl(state):
        """(win, draw, loss) from Barça's side, plus the features as Barça - opponent."""
        if venue == "home":
            (lose, draw, win), x = probabilities(model, state, BARCA, rival, comp, when)
        elif venue == "away":
            (win, draw, lose), x = probabilities(model, state, rival, BARCA, comp, when)
            x = {k: -v if k != "is_champions" else v for k, v in x.items()}
        else:
            # Neutral ground: play the match twice, once with each team "at home", and average.
            # That cancels the home advantage the model learned.
            (l1, d1, w1), x = probabilities(model, state, BARCA, rival, comp, when)
            (w2, d2, l2), _ = probabilities(model, state, rival, BARCA, comp, when)
            win, draw, lose = (w1 + w2) / 2, (d1 + d2) / 2, (l1 + l2) / 2
        return (win, draw, lose), x

    (win, draw, lose), x = wdl(state)
    # Same match again, with each team's Elo lowered for its missing players.
    r = state["ratings"]
    weakened = {**r, BARCA: r[BARCA] - penalty(absent_barca), rival: r[rival] - penalty(absent_rival)}
    (win2, draw2, lose2), _ = wdl({**state, "ratings": weakened})

    played = df[(df.home == BARCA) | (df.away == BARCA)]
    h2h = played[(played.home == rival) | (played.away == rival)]

    def line(m):
        letter = "D" if m.hg == m.ag else "W" if (m.home == BARCA) == (m.hg > m.ag) else "L"
        return f"  {m.date:%d/%m/%Y}  {letter}  {m.home} {m.hg}-{m.ag} {m.away}  ({m.comp})"

    def letters(ms):
        return " ".join(line(m).split()[1] for m in ms.itertuples())

    where = {"home": "home", "away": "away", "neutral": "neutral venue"}[venue]
    print(f"\n=== Barcelona vs {rival} · {comp} · {where} · {when:%d/%m/%Y} ===")
    print(f"\nLast match\n{line(played.iloc[-1])}")
    print(f"\nForm (last 5): {letters(played.tail(5))}")
    for m in played.tail(5).itertuples():
        print(line(m))
    print(f"\nHead-to-head vs {rival} since 22/23: {letters(h2h) or 'no matches in the data'}")
    for m in h2h.itertuples():
        print(line(m))
    print(f"\nElo   Barcelona {r[BARCA]:.0f}   {rival} {r[rival]:.0f}   (difference {r[BARCA] - r[rival]:+.0f})")
    print(f"Form, last 5 (Barça - opponent): points/match {x['form_pts_diff']:+.2f}, goal diff/match {x['form_gd_diff']:+.2f}")
    print(f"Rest days (Barça - opponent): {x['rest_diff']:+.0f}")
    print(f"\nMODEL PREDICTION\n  Barça win   {win:6.1%}\n  Draw        {draw:6.1%}\n  Barça loss  {lose:6.1%}")
    if absent_barca or absent_rival:
        for team, absent in [(BARCA, absent_barca), (rival, absent_rival)]:
            if absent:
                names = ", ".join(f"{n} ({t})" for n, t in absent.items())
                print(f"\nAbsent, {team}: {names}  →  Elo {r[team]:.0f} → {weakened[team]:.0f}")
        print(f"\nWITH ABSENCES\n  Barça win   {win2:6.1%}\n  Draw        {draw2:6.1%}\n  Barça loss  {lose2:6.1%}")
    print()

    new = not LOG.exists()
    with LOG.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["predicted_on", "match_date", "rival", "comp", "venue", "p_win", "p_draw", "p_loss",
                        "adj_win", "adj_draw", "adj_loss", "absent"])
        absent = "; ".join(f"{n} ({t})" for n, t in {**absent_barca, **absent_rival}.items())
        w.writerow([date.today(), when.date(), rival, comp, venue, round(win, 4), round(draw, 4), round(lose, 4),
                    round(win2, 4), round(draw2, 4), round(lose2, 4), absent])


def match_date(rival, comp, text):
    """Empty date: take it from the Champions League calendar if the fixture is there, otherwise today."""
    if text:
        return pd.to_datetime(text, dayfirst=True)
    if comp == "champions":
        up = data.upcoming_champions()
        up = up[((up.home == BARCA) & (up.away == rival)) | ((up.home == rival) & (up.away == BARCA))]
        if len(up):
            return up.date.min()
    return pd.Timestamp(date.today())


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CHANGE ONLY THESE 4 LINES, THEN RUN THE FILE                        ║
# ╚══════════════════════════════════════════════════════════════════════╝
OPPONENT = "Galatasaray"       # team name (if you misspell it, you get suggestions)
COMPETITION = "champions"      # liga | champions | copa | supercopa
VENUE = "away"            # home | away | neutral
DATE = ""       # DD/MM/YYYY. Empty = date from the Champions League calendar, or today
# Missing players, optional. Type: star | starter | rotation. Leave {} if nobody is missing.
ABSENT_BARCA = {"Raphinha": "star"}
ABSENT_OPPONENT = {}

if __name__ == "__main__":
    venue, comp = VENUE.lower().strip(), COMPETITION.lower().strip()
    if venue not in ["home", "away", "neutral"]:
        sys.exit(f"VENUE must be home, away or neutral (you wrote '{VENUE}')")
    if comp not in ["liga", "champions", "copa", "supercopa"]:
        sys.exit(f"COMPETITION must be liga, champions, copa or supercopa (you wrote '{COMPETITION}')")
    predict(OPPONENT.strip(), comp, venue, match_date(OPPONENT.strip(), comp, DATE.strip()), ABSENT_BARCA, ABSENT_OPPONENT)
