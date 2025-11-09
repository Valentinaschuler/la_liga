# src/quiniela/model.py
import joblib
from sklearn.ensemble import RandomForestClassifier

class QuinielaModel:
    def __init__(self, model=None):
        self.model = model

    def train(self, X, y, n_estimators=300, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs
        )
        self.model.fit(X, y)
        return self.model

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
