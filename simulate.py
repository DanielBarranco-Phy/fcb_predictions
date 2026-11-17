"""Simulate the rest of the La Liga season 10,000 times with the model.

    python simulate.py

For every match still to play, the model gives win / draw / loss probabilities. We roll the dice for
each match, add up the points, and repeat 10,000 times. Counting how often each team finishes first
gives a title probability. Simplification: ratings and form are frozen at today's values.
"""
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
import model as mdl
from predict import probabilities

N = 10_000
FIXTURES = "https://fixturedownload.com/feed/json/la-liga-2026"
NAMES = {  # fixturedownload.com -> football-data.co.uk
    "Athletic Club": "Ath Bilbao", "Atlético de Madrid": "Ath Madrid", "CA Osasuna": "Osasuna", "Celta": "Celta",
    "Deportivo Alavés": "Alaves", "Elche CF": "Elche", "FC Barcelona": "Barcelona", "Getafe CF": "Getafe",
    "Levante UD": "Levante", "Málaga CF": "Malaga", "R. Racing Club": "Santander", "RC Deportivo": "La Coruna",
    "RCD Espanyol de Barcelona": "Espanol", "Rayo Vallecano": "Vallecano", "Real Betis": "Betis", "Real Madrid": "Real Madrid",
    "Real Sociedad": "Sociedad", "Sevilla FC": "Sevilla", "Valencia CF": "Valencia", "Villarreal CF": "Villarreal",
}


def main():
    model, df, state = mdl.train()
    season = df[(df.comp == "liga") & (df.date >= "2026-07-01")]
    teams = sorted(NAMES.values())
    points = {t: 0 for t in teams}
    for m in season.itertuples():
        points[m.home] += 3 if m.hg > m.ag else 1 if m.hg == m.ag else 0
        points[m.away] += 3 if m.ag > m.hg else 1 if m.hg == m.ag else 0
    played = set(zip(season.home, season.away))

    req = urllib.request.Request(FIXTURES, headers={"User-Agent": "Mozilla/5.0"})
    fixtures = json.load(urllib.request.urlopen(req, timeout=60))
    left = [(NAMES[f["HomeTeam"]], NAMES[f["AwayTeam"]], pd.to_datetime(f["DateUtc"]).tz_localize(None)) for f in fixtures]
    left = [(h, a, d) for h, a, d in left if (h, a) not in played]

    probs = np.array([probabilities(model, state, h, a, "liga", d)[0] for h, a, d in left])  # [away, draw, home]
    rng = np.random.default_rng(2026)
    roll = rng.random((N, len(left)))
    away_win, draw = roll < probs[:, 0], (roll >= probs[:, 0]) & (roll < probs[:, 0] + probs[:, 1])
    home_win = ~away_win & ~draw

    idx = {t: i for i, t in enumerate(teams)}
    total = np.tile([points[t] for t in teams], (N, 1)).astype(float)
    for j, (h, a, _) in enumerate(left):
        total[:, idx[h]] += 3 * home_win[:, j] + draw[:, j]
        total[:, idx[a]] += 3 * away_win[:, j] + draw[:, j]
    total += rng.random(total.shape) * 0.01  # break ties at random
    rank = (-total).argsort(1).argsort(1) + 1

    out = pd.DataFrame({
        "team": teams,
        "points_now": [points[t] for t in teams],
        "avg_points": total.mean(0).round(1),
        "title_%": ((rank == 1).mean(0) * 100).round(1),
        "top4_%": ((rank <= 4).mean(0) * 100).round(1),
        "relegation_%": ((rank >= 18).mean(0) * 100).round(1),
    }).sort_values("avg_points", ascending=False)
    print(f"\n{len(season)} matches played, {len(left)} left, {N:,} simulated seasons\n")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
