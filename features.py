# src/quiniela/features.py
import pandas as pd
import numpy as np

def prepare_features(df):
    """
    df: matches_features.csv loaded DataFrame that already contains
    rolling stats columns. This function selects & returns X with the same features used during training.
    """
    # Ensure date is datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # Basic defense: fill NA with reasonable defaults
    fill_val = 1.5
    df = df.fillna(fill_val)
    FEATURES = [
        "home_team_avg_goals_for", "home_team_avg_goals_against",
        "away_team_avg_goals_for", "away_team_avg_goals_against",
        "avg_goals_for_combined", "avg_goals_against_combined",
        "year", "month", "weekday"
    ]
    X = df[FEATURES].copy()
    return X
