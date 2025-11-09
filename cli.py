# src/quiniela/cli.py
import click
import sqlite3
from datetime import datetime
import pandas as pd
import os

from .model import QuinielaModel
from .data import load_features_from_csv, load_matches_for_season
from .features import prepare_features

MODEL_PATH = "models/my_quiniela.model"
DB_PATH = "laliga.sqlite"

@click.group()
def cli():
    pass

@cli.command()
@click.option("--training_seasons", default="2010:2020", help="Seasons range like 2010:2020")
def train(training_seasons):
    # parse training seasons (start:end)
    df = load_features_from_csv()
    # simple parse: use seasons as strings; if your 'season' column sorts lexicographically, filter
    start, end = training_seasons.split(":")
    train_df = df[df["season"] >= f"{start}-{int(start)+1}"]  # adapt if your format differs
    X = prepare_features(train_df)
    y = train_df["target"].astype(int)
    qm = QuinielaModel()
    qm.train(X, y)
    os.makedirs("models", exist_ok=True)
    qm.save(MODEL_PATH)
    click.echo(f"Model saved to {MODEL_PATH}")

@cli.command()
@click.argument("season")
@click.argument("division", type=int)
@click.argument("matchday", type=int)
def predict(season, division, matchday):
    # load model
    if not os.path.exists(MODEL_PATH):
        raise click.ClickException("Model not found. Run quiniela train first.")
    qm = QuinielaModel()
    qm.load(MODEL_PATH)

    # load matches to predict for that season/division/matchday from DB
    conn = sqlite3.connect(DB_PATH)
    df_matches = load_matches_for_season(conn, season, division, matchday)
    if df_matches.empty:
        click.echo("No matches found for that season/division/matchday")
        return

    # You need to build features for these matches consistent with training.
    # For simplicity, this example expects you have matches_features.csv containing rows for these matches.
    feat_df = pd.read_csv("reports/matches_features.csv", parse_dates=["date"])
    mask = (feat_df["season"] == season) & (feat_df["division"] == division) & (feat_df["matchday"] == matchday)
    X_pred_df = feat_df[mask].copy()
    if X_pred_df.empty:
        click.echo("No prepared feature rows for this matchday. Run notebook to compute them.")
        return

    X = prepare_features(X_pred_df)
    preds = qm.predict(X)
    probs = qm.predict_proba(X) if hasattr(qm.model, "predict_proba") else None

    # Map numeric labels to '1','X','2'
    label_map = {0: "1", 1: "X", 2: "2"}
    rows = []
    ts = datetime.utcnow().isoformat()
    for i, r in X_pred_df.iterrows():
        pred_label = label_map[int(preds[i])]
        conf = float(probs[i].max()) if probs is not None else 0.5
        rows.append((season, ts, division, matchday, r["home_team"], r["away_team"], pred_label, conf))

    # Insert into DB Predictions table
    conn.executemany(
        "INSERT INTO Predictions (season, timestamp, division, matchday, home_team, away_team, prediction, confidence) VALUES (?,?,?,?,?,?,?,?)",
        rows
    )
    conn.commit()
    conn.close()
    click.echo(f"Inserted {len(rows)} predictions into DB.")
