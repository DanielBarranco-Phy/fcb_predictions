"""Turn the match table into the numbers the model sees. Every feature uses only information from BEFORE kickoff."""
from collections import defaultdict

import pandas as pd

import elo

FEATURES = ["elo_diff", "form_pts_diff", "form_gd_diff", "form_sot_diff", "rest_diff", "is_champions"]
FORM_N = 5
MAX_REST = 10  # beyond ~10 days extra rest stops mattering; also tames the summer break


def team_form(history, date):
    """history: list of (date, points, goal_diff, shots_on_target_diff) for one team, oldest first.
    Shots are only known for league matches (None in Europe and cups), so shot form uses the last
    FORM_N matches that have them."""
    last = history[-FORM_N:]
    if not last:
        return 1.4, 0.0, 0.0, MAX_REST  # no matches yet: assume an average team
    pts = sum(m[1] for m in last) / len(last)
    gd = sum(m[2] for m in last) / len(last)
    shots = [m[3] for m in history[-3 * FORM_N:] if m[3] is not None][-FORM_N:]
    sot = sum(shots) / len(shots) if shots else 0.0
    rest = min((date - last[-1][0]).days, MAX_REST)
    return pts, gd, sot, rest


def row(date, comp, elo_h, elo_a, hist_h, hist_a):
    ph, gh, sh, rh = team_form(hist_h, date)
    pa, ga, sa, ra = team_form(hist_a, date)
    return {
        "elo_diff": elo_h - elo_a,
        "form_pts_diff": ph - pa,
        "form_gd_diff": gh - ga,
        "form_sot_diff": sh - sa,
        "rest_diff": rh - ra,
        "is_champions": int(comp == "champions"),
    }


def build(matches, bases=None):
    """Returns (matches + feature columns, state). `state` is everything needed to predict a future match."""
    groups = elo.team_groups(matches)
    bases = bases or elo.fit_bases(matches, groups)
    elo_h, elo_a, ratings = elo.run(matches, bases, groups)
    history = defaultdict(list)
    rows = []
    hst = matches.hst if "hst" in matches else [None] * len(matches)
    ast = matches.ast if "ast" in matches else [None] * len(matches)
    for (d, comp, h, a, hg, ag, sh, sa), eh, ea in zip(
        zip(matches.date, matches.comp, matches.home, matches.away, matches.hg, matches.ag, hst, ast), elo_h, elo_a
    ):
        rows.append(row(d, comp, eh, ea, history[h], history[a]))
        sot = None if pd.isna(sh) or pd.isna(sa) else sh - sa
        history[h].append((d, 3 if hg > ag else 1 if hg == ag else 0, hg - ag, sot))
        history[a].append((d, 3 if ag > hg else 1 if hg == ag else 0, ag - hg, None if sot is None else -sot))
    out = matches.copy()
    out["elo_home"], out["elo_away"] = elo_h, elo_a
    for f in FEATURES:
        out[f] = [r[f] for r in rows]
    return out, {"ratings": ratings, "history": history, "bases": bases, "groups": groups}
