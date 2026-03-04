"""
Training and cross-validated evaluation for score prediction models.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from kinescope.prediction.models import get_model, PARAM_GRIDS
from kinescope.prediction.evaluate import plot_roc, plot_pr_curve, plot_confusion_matrix, plot_feature_importance


def load_features_and_labels(
    features_csv: str,
    scores_csv: str,
    video_col: str = "video",
    score_col: str = "score",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load and merge feature matrix with score labels.

    Parameters
    ----------
    features_csv : str
        Path to consolidated feature CSV (output of PoseProcessingPipeline).
    scores_csv : str
        CSV with at minimum two columns: video identifier and score.
    video_col : str
        Column name for the video/subject identifier.
    score_col : str
        Column name for the score/label.

    Returns
    -------
    (X, y) where X is feature DataFrame and y is label Series, aligned by video_col.
    """
    features = pd.read_csv(features_csv)
    scores = pd.read_csv(scores_csv)

    # Flexible merge: try common identifier column names
    score_id_candidates = [video_col, "infant", "gma_id", "subject", "id"]
    merge_col = next(
        (c for c in score_id_candidates if c in scores.columns), None
    )
    if merge_col is None:
        raise ValueError(
            f"Could not find a subject ID column in scores CSV. "
            f"Expected one of: {score_id_candidates}. Found: {list(scores.columns)}"
        )

    merged = pd.merge(
        features,
        scores[[merge_col, score_col]].rename(columns={merge_col: video_col}),
        on=video_col,
        how="inner",
    ).dropna()

    X = merged.drop(columns=[video_col, score_col])
    y = merged[score_col]
    return X, y


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "xgboost",
    test_size: float = 0.2,
    splits_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    cv_folds: int = 5,
    binary: bool = True,
    random_state: int = 42,
) -> dict:
    """
    Train a model with GridSearchCV and evaluate on a held-out test set.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (samples × features). NaN columns are dropped.
    y : pd.Series
        Labels (integer or categorical scores).
    model_name : str
        Model type. One of: logreg, xgboost, random_forest, vit.
    test_size : float
        Fraction of data for test split (used if splits_dir is None).
    splits_dir : str, optional
        Directory containing train.csv, val.csv, test.csv with a 'video' column.
        If provided, overrides test_size.
    output_dir : str, optional
        Directory to save evaluation plots and best model.
    cv_folds : int
        Number of StratifiedKFold folds for GridSearchCV.
    binary : bool
        If True, binarize labels: 0 = lowest score class, 1 = all others.
    random_state : int

    Returns
    -------
    dict with keys:
        best_model, best_params, test_predictions, test_probas,
        metrics (dict: balanced_accuracy, roc_auc, avg_precision)
    """
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score, average_precision_score

    # Drop constant / all-NaN columns
    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.std() > 0]
    X = X.fillna(X.median())

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    if binary:
        y_enc = (y_enc > 0).astype(int)

    # Train / test split
    if splits_dir is not None:
        splits_dir = Path(splits_dir)
        train_ids = pd.read_csv(splits_dir / "train.csv").iloc[:, 0].astype(str).tolist()
        test_ids  = pd.read_csv(splits_dir / "test.csv").iloc[:, 0].astype(str).tolist()
        X_train = X.loc[X.index.isin(train_ids)]
        y_train = pd.Series(y_enc, index=X.index).loc[X.index.isin(train_ids)]
        X_test  = X.loc[X.index.isin(test_ids)]
        y_test  = pd.Series(y_enc, index=X.index).loc[X.index.isin(test_ids)]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=test_size, stratify=y_enc, random_state=random_state
        )

    # GridSearchCV
    base_model = get_model(model_name)
    param_grid = PARAM_GRIDS.get(model_name, {})

    if param_grid:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        grid_search = GridSearchCV(
            base_model, param_grid,
            cv=cv, scoring="balanced_accuracy",
            n_jobs=-1, refit=True, verbose=0,
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
    else:
        best_model = base_model.fit(X_train, y_train)
        best_params = {}

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    y_proba = (best_model.predict_proba(X_test)[:, 1]
               if hasattr(best_model, "predict_proba") else y_pred.astype(float))

    metrics = {
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba) if binary else None,
        "avg_precision": average_precision_score(y_test, y_proba) if binary else None,
    }

    print(f"Model: {model_name}")
    print(f"Best params: {best_params}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.3f}")
    if binary:
        print(f"ROC AUC:           {metrics['roc_auc']:.3f}")
        print(f"Avg precision:     {metrics['avg_precision']:.3f}")

    # Save evaluation plots
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        plot_roc(y_test, y_proba, save_path=out / "roc_curve.png")
        plot_pr_curve(y_test, y_proba, save_path=out / "pr_curve.png")
        plot_confusion_matrix(y_test, y_pred, save_path=out / "confusion_matrix.png")
        if hasattr(best_model, "feature_importances_"):
            plot_feature_importance(
                best_model.feature_importances_, X_train.columns,
                save_path=out / "feature_importance.png",
            )
        elif hasattr(best_model, "coef_"):
            plot_feature_importance(
                np.abs(best_model.coef_[0]), X_train.columns,
                save_path=out / "feature_importance.png",
                title="Feature Coefficients (|coef|)",
            )

        # Save predictions
        pd.DataFrame({
            "y_true": y_test, "y_pred": y_pred, "y_proba": y_proba
        }).to_csv(out / "predictions.csv", index=False)

        # Save model
        import pickle
        with open(out / f"{model_name}_best_model.pkl", "wb") as f:
            pickle.dump(best_model, f)

    return {
        "best_model": best_model,
        "best_params": best_params,
        "test_predictions": y_pred,
        "test_probas": y_proba,
        "metrics": metrics,
    }
