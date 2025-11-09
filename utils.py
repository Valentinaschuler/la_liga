import pandas as pd

def parse_score(score):
    if not isinstance(score, str) or ':' not in score:
        return pd.Series([None, None])
    try:
        h, a = score.split(':')
        return pd.Series([int(h), int(a)])
    except ValueError:
        return pd.Series([None, None])
