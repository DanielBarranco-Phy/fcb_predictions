"""Refresh the data after each matchday, then show the model's track record.

    python update.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
import data

LOG = Path(__file__).parent / "predictions.csv"
HISTORY = Path(__file__).parent / "track_record.csv"

data.download()
m = data.load_matches()
barca = m[(m.home == "Barcelona") | (m.away == "Barcelona")]
print(f"{len(m)} matches loaded. Barça's latest:")
print(barca.tail(3)[["date", "comp", "home", "hg", "ag", "away"]].to_string(index=False))

if LOG.exists():
    names = {"p_win": "Win", "p_draw": "Draw", "p_loss": "Loss"}
    rows = []
    for p in pd.read_csv(LOG, parse_dates=["predicted_on", "match_date"]).itertuples():
        pick = max(names, key=lambda k: getattr(p, k))
        adj = {"p_win": p.adj_win, "p_draw": p.adj_draw, "p_loss": p.adj_loss}
        pick2 = max(adj, key=adj.get)
        row = {"match": p.match_date, "opponent": p.rival, "comp": p.comp,
               "prediction": f"{names[pick]} ({getattr(p, pick):.0%})", "with_absences": f"{names[pick2]} ({adj[pick2]:.0%})", "score": "", "actual": "", "status": "pending",
               "status_abs": "pending"}
        # the match this prediction was about: Barça's first game against that rival, same venue, after predicting
        home = {"home": "Barcelona", "away": p.rival}.get(p.venue)
        g = barca[((barca.home == p.rival) | (barca.away == p.rival)) & (barca.comp == p.comp)
                  & (barca.date >= p.predicted_on) & ((barca.home == home) | (home is None))]
        if len(g):
            g = g.iloc[0]
            real = "p_draw" if g.hg == g.ag else "p_win" if (g.home == "Barcelona") == (g.hg > g.ag) else "p_loss"
            row.update(match=g.date, score=f"{g.home} {g.hg}-{g.ag} {g.away}", actual=names[real],
                       status="HIT" if pick == real else "MISS", status_abs="HIT" if pick2 == real else "MISS")
        rows.append(row)
    # predicted the same match several times: only the last prediction counts
    h = pd.DataFrame(rows).drop_duplicates(["match", "opponent", "comp"], keep="last").sort_values("match")
    h["match"] = h.match.dt.strftime("%d/%m/%Y")
    h.to_csv(HISTORY, index=False)
    hits, misses = (h.status == "HIT").sum(), (h.status == "MISS").sum()
    print(f"\nMODEL TRACK RECORD\n{h.to_string(index=False)}")
    print(f"\nHits: {hits}   Misses: {misses}   Pending: {(h.status == 'pending').sum()}", end="")
    print(f"   →  {hits}/{hits + misses} ({hits / (hits + misses):.0%})" if hits + misses else "")
    hits2 = (h.status_abs == "HIT").sum()
    print(f"With absences: {hits2}/{hits + misses}" if hits + misses else "")
