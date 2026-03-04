"""
Sklearn-compatible model factories for score prediction.

Supported models: logreg, xgboost, random_forest, vit

The ViT model requires the [vit] extra: pip install kinescope[vit]
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

SUPPORTED_MODELS = ["logreg", "xgboost", "random_forest", "vit"]

# Parameter grids for GridSearchCV
PARAM_GRIDS = {
    "logreg": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "solver": ["lbfgs"],
        "max_iter": [1000],
    },
    "xgboost": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.1, 0.05],
    },
    "random_forest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_leaf": [1, 3],
    },
    "vit": {
        "lr": [1e-3, 1e-4],
        "epochs": [50, 100],
    },
}


def get_model(name: str):
    """
    Return an unfitted sklearn-compatible estimator for the given model name.

    Parameters
    ----------
    name : str
        One of: "logreg", "xgboost", "random_forest", "vit"

    Returns
    -------
    sklearn BaseEstimator (or compatible)
    """
    name = name.lower()
    if name == "logreg":
        return LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )
    elif name == "xgboost":
        return XGBClassifier(
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
    elif name == "random_forest":
        return RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    elif name == "vit":
        from kinescope.prediction._vit import PoseViTClassifier
        return PoseViTClassifier()
    else:
        raise ValueError(f"Unknown model '{name}'. Choose from: {SUPPORTED_MODELS}")
