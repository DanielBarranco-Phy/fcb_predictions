"""Elo ratings: one number per team that says how strong it is right now.

Before a match:  expected = 1 / (1 + 10 ** (-(elo_home + HOME_ADV - elo_away) / 400))
After a match:   elo_home += K * margin * (actual - expected)      (and the away team loses the same)
where actual is 1 for a win, 0.5 for a draw, 0 for a loss.
"""
from collections import defaultdict

K = 20          # how fast ratings react to one result
HOME_ADV = 60   # Elo points the home team is worth just for playing at home
START = 1500    # La Liga's starting average; other leagues start higher or lower (see fit_bases)
LOWER_GAP = 300  # lower-division cup rivals start this far below La Liga


def expected(elo_home, elo_away, neutral=0):
    return 1 / (1 + 10 ** (-(elo_home + HOME_ADV * (1 - neutral) - elo_away) / 400))


def _margin(goal_diff):
    """Bigger wins move ratings more, with diminishing returns (same idea as ClubElo / World Football Elo)."""
    g = abs(goal_diff)
    return 1 if g <= 1 else 1.5 if g == 2 else (11 + g) / 8


def team_groups(matches):
    """Each team belongs to its domestic league; teams we only see in Europe or in cups get their own group."""
    groups = {}
    for league in ["domestic", "UCL", "CUP"]:
        sub = matches[~matches.league.isin(["UCL", "CUP"])] if league == "domestic" else matches[matches.league == league]
        for h, a, lg in zip(sub.home, sub.away, sub.league):
            name = {"UCL": "OTHER", "CUP": "LOWER"}.get(lg, lg)
            groups.setdefault(h, name)
            groups.setdefault(a, name)
    return groups


def run(matches, bases, groups):
    """Walk through every match in date order. Returns each team's rating BEFORE each match, and the final ratings."""
    ratings = defaultdict(lambda: START)
    for team, g in groups.items():
        ratings[team] = bases.get(g, START)
    before_h, before_a = [], []
    for h, a, hg, ag, n in zip(matches.home, matches.away, matches.hg, matches.ag, matches.neutral):
        rh, ra = ratings[h], ratings[a]
        before_h.append(rh)
        before_a.append(ra)
        actual = 1 if hg > ag else 0.5 if hg == ag else 0
        change = K * _margin(hg - ag) * (actual - expected(rh, ra, n))
        ratings[h], ratings[a] = rh + change, ra - change
    return before_h, before_a, dict(ratings)


def fit_bases(matches, groups, rounds=30):
    """Find each league's starting level from Champions League results.

    Inside one league Elo is zero-sum, so without this the Turkish champion would look as strong as
    the Spanish one. We repeat: run Elo, see which leagues did better in Europe than their ratings
    expected, raise their starting level, lower the others.
    """
    bases = {g: START for g in set(groups.values())}
    bases["LOWER"] = START - LOWER_GAP
    for _ in range(rounds):
        bh, ba, _ = run(matches, bases, groups)
        surprise, count = defaultdict(float), defaultdict(int)
        for h, a, hg, ag, n, rh, ra in zip(matches.home, matches.away, matches.hg, matches.ag, matches.neutral, bh, ba):
            gh, ga = groups[h], groups[a]
            if gh == ga or "LOWER" in (gh, ga):
                continue
            s = (1 if hg > ag else 0.5 if hg == ag else 0) - expected(rh, ra, n)
            surprise[gh] += s; count[gh] += 1
            surprise[ga] -= s; count[ga] += 1
        for g in count:
            bases[g] += 150 * surprise[g] / count[g]
        shift = START - bases["SP1"]  # keep La Liga anchored at 1500 so numbers are comparable between runs
        bases = {g: b + shift if g != "LOWER" else b for g, b in bases.items()}
    return bases
