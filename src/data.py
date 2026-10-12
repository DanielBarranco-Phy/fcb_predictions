"""Download raw data and load every match into one table.

Sources
- football-data.co.uk : domestic leagues (results + betting odds), one CSV per season
- fixturedownload.com : UEFA Champions League results and upcoming fixtures
- ESPN (site.api.espn.com) : Barça's Copa del Rey and Supercopa matches (unofficial but public API)
- manual_results.csv : results typed by hand while a source has not published them yet
"""
import json
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

SEASONS = ["2223", "2324", "2425", "2526", "2627"]
CURRENT = SEASONS[-1]
# La Liga is what we model; the other leagues only exist to rate Barça's European rivals.
LEAGUES = ["SP1", "E0", "D1", "I1", "F1", "P1", "N1", "B1", "T1", "G1", "SC0"]

# Champions League / ESPN name -> football-data.co.uk name (only where they differ)
ALIASES = {
    "AEK Athens": "AEK", "Athletic Club": "Ath Bilbao", "Atleti": "Ath Madrid", "Atlético": "Ath Madrid",
    "Atlético de Madrid": "Ath Madrid", "B. Dortmund": "Dortmund", "Bayern": "Bayern Munich",
    "Bayern München": "Bayern Munich", "Braga": "Sp Braga", "Crvena zvezda": "Crvena Zvezda",
    "FC Porto": "Porto", "Fenerbahçe": "Fenerbahce", "Frankfurt": "Ein Frankfurt", "GNK Dinamo": "Dinamo Zagreb",
    "Leipzig": "RB Leipzig", "Man Utd": "Man United", "Olympiacos": "Olympiakos", "PSV": "PSV Eindhoven",
    "Paris": "Paris SG", "Real Betis": "Betis", "Real Sociedad": "Sociedad", "Shakhtar": "Shakhtar Donetsk",
    "Sporting CP": "Sp Lisbon", "Union SG": "St. Gilloise",
    "Atlético Madrid": "Ath Madrid", "Alavés": "Alaves", "Celta Vigo": "Celta", "Deportivo": "La Coruna",
    "Espanyol": "Espanol", "Málaga": "Malaga", "Racing Santander": "Santander", "Rayo Vallecano": "Vallecano",
    "Real Oviedo": "Oviedo", "Leganés": "Leganes", "Cádiz": "Cadiz", "Almería": "Almeria", "Real Valladolid": "Valladolid",
}
ESPN_BARCA = 83
CUPS = {"copa": "esp.copa_del_rey", "supercopa": "esp.super_cup"}


def _fetch(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        dest.write_bytes(r.read())


def download():
    """Past seasons never change, so after the first run only the current one is refreshed."""
    RAW.mkdir(parents=True, exist_ok=True)
    for s in SEASONS:
        year = "20" + s[:2]
        files = {f"{lg}_{s}.csv": f"https://www.football-data.co.uk/mmz4281/{s}/{lg}.csv" for lg in LEAGUES}
        files[f"UCL_{year}.json"] = f"https://fixturedownload.com/feed/json/champions-league-{year}"
        for comp, code in CUPS.items():
            files[f"CUP_{comp}_{year}.json"] = (
                f"https://site.api.espn.com/apis/site/v2/sports/soccer/{code}/teams/{ESPN_BARCA}/schedule?season={year}")
        for name, url in files.items():
            dest = RAW / name
            if s != CURRENT and dest.exists() and dest.stat().st_size > 0:
                continue
            try:
                _fetch(url, dest)
            except Exception as e:  # keep the old copy if a source is temporarily down
                print(f"  ! could not download {name}: {e}")


def _leagues():
    frames = []
    for f in sorted(RAW.glob("*_*.csv")):
        league = f.stem.split("_")[0]
        df = pd.read_csv(f, encoding="utf-8-sig", on_bad_lines="skip")
        df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        frames.append(pd.DataFrame({
            "date": pd.to_datetime(df["Date"], dayfirst=True),
            "comp": "liga" if league == "SP1" else "other_league",
            "league": league,
            "home": df["HomeTeam"].str.strip(),
            "away": df["AwayTeam"].str.strip(),
            "hg": df["FTHG"].astype(int),
            "ag": df["FTAG"].astype(int),
            "neutral": 0,
            # market-average odds: used only as a benchmark, never as a model input
            "odds_h": df.get("AvgH"), "odds_d": df.get("AvgD"), "odds_a": df.get("AvgA"),
        }))
    return pd.concat(frames, ignore_index=True)


def _champions():
    played, upcoming = [], []
    for f in sorted(RAW.glob("UCL_*.json")):
        rows = json.loads(f.read_text())
        last_round = max(r["RoundNumber"] for r in rows)
        final_is_single = sum(r["RoundNumber"] == last_round for r in rows) == 1
        for r in rows:
            m = {
                "date": pd.to_datetime(r["DateUtc"]).tz_localize(None).normalize(),
                "comp": "champions", "league": "UCL",
                "home": ALIASES.get(r["HomeTeam"], r["HomeTeam"]),
                "away": ALIASES.get(r["AwayTeam"], r["AwayTeam"]),
                "neutral": int(final_is_single and r["RoundNumber"] == last_round),
            }
            if r["HomeTeamScore"] is None:
                upcoming.append(m)
            else:
                played.append({**m, "hg": r["HomeTeamScore"], "ag": r["AwayTeamScore"]})
    return pd.DataFrame(played), pd.DataFrame(upcoming)


def _cups():
    rows = []
    for f in sorted(RAW.glob("CUP_*.json")):
        comp = f.stem.split("_")[1]
        for e in json.loads(f.read_text()).get("events", []):
            c = e["competitions"][0]
            status = c["status"]["type"]
            if not status["completed"]:
                continue
            t = {x["homeAway"]: x for x in c["competitors"]}
            hg, ag = (int(float(t[k]["score"]["value"])) for k in ("home", "away"))
            if status["name"] in ("STATUS_FINAL_AET", "STATUS_FINAL_PEN"):
                # We predict the 90-minute result (same rule as bookmakers), and a match that needed
                # extra time was a draw after 90. ESPN only gives the final score, so level it.
                hg = ag = min(hg, ag)
            name = lambda k: ALIASES.get(t[k]["team"]["displayName"], t[k]["team"]["displayName"])
            rows.append({
                "date": pd.to_datetime(e["date"]).tz_localize(None).normalize(),
                "comp": comp, "league": "CUP", "home": name("home"), "away": name("away"), "hg": hg, "ag": ag,
                # the Supercopa is played abroad and the Copa final at a neutral stadium
                "neutral": int(comp == "supercopa" or e.get("seasonType", {}).get("name") == "Final"),
            })
    return pd.DataFrame(rows)


def _manual(official):
    """Results typed by hand in manual_results.csv, for when a source has not published a match yet.
    Once the official source has the match, the manual line is ignored."""
    f = ROOT / "manual_results.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f, comment="#", skipinitialspace=True)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    for c in ["comp", "home", "away"]:
        df[c] = df[c].str.strip()
    df["league"] = df.comp.map({"liga": "SP1", "champions": "UCL"}).fillna("CUP")
    df["neutral"] = (df.comp == "supercopa").astype(int)
    known = set(official.home) | set(official.away)
    for t in set(df.home) | set(df.away):
        if t not in known:
            print(f"  ! manual_results.csv: unknown team '{t}' (check the spelling)")
    already = [((official.home == r.home) & (official.away == r.away) & ((official.date - r.date).abs().dt.days <= 3)).any()
               for r in df.itertuples()]
    return df[[not x for x in already]]


def load_matches():
    """All played matches, oldest first, one row per match."""
    df = pd.concat([_leagues(), _champions()[0], _cups()], ignore_index=True)
    df = pd.concat([df, _manual(df)], ignore_index=True)
    df["result"] = "D"
    df.loc[df.hg > df.ag, "result"] = "H"
    df.loc[df.hg < df.ag, "result"] = "A"
    return df.sort_values("date", kind="stable").reset_index(drop=True)


def upcoming_champions():
    return _champions()[1]


if __name__ == "__main__":
    download()
    m = load_matches()
    print(m.groupby("comp").size())
