# src/quiniela/data.py
import pandas as pd
import sqlite3

def load_features_from_csv(path="reports/matches_features.csv", parse_dates=["date"]):
    return pd.read_csv(path, parse_dates=parse_dates)

def load_matches_for_season(conn, season, division=1, matchday=None):
    # conn is sqlite3.Connection or path
    if isinstance(conn, str):
        conn = sqlite3.connect(conn)
    q = "SELECT season, division, matchday, date, home_team, away_team, score FROM Matches WHERE season=? AND division=?"
    params = [season, division]
    if matchday is not None:
        q += " AND matchday=?"
        params.append(matchday)
    df = pd.read_sql(q, conn, params=params, parse_dates=["date"])
    return df
