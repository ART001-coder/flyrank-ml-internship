"""
Week 5-7 lane work: Refresh / Content Opportunity Scoring.

Differs from the reference pipeline (scripts/03_train_model.py) in three ways:
1. Features: adds peer-relative signals (z-scores within content_type and
   position_tier) instead of only raw/log aggregates.
2. Validation: repeated grouped stratified K-fold (5 folds x 3 repeats,
   grouped by client_id) instead of one client-holdout split, so metrics
   are reported as mean +/- std rather than a single number.
3. Output: a continuous, demand-weighted "opportunity_score" (decline
   probability x log(impressions)) for ranking, not just a probability.

Leakage guard: trend_direction and trend_pct (and the last_30d/prev_30d
columns they're computed from) are never used as features -- only 90-day
aggregates, content properties, and keyword context go in.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "content_refresh_anonymized.csv"
OUT_DIR = ROOT / "work" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

BASE_NUMERIC = [
    "search_volume", "competition", "cpc", "word_count", "char_count",
    "days_with_impressions", "days_with_sessions", "content_age_days",
    "days_since_last_update", "ctr", "avg_position", "engagement_rate",
    "scroll_rate", "ai_traffic_pct",
]
LOG_SOURCE = ["impressions_90d", "clicks_90d", "sessions_90d", "ai_sessions_90d"]
CATEGORICAL = [
    "competition_level", "content_type", "main_intent", "age_tier",
    "freshness_tier", "word_count_tier", "impression_tier", "position_tier",
]
PEER_GROUP_COLS = ["content_type", "position_tier"]
PEER_TARGET_COLS = ["ctr", "engagement_rate", "avg_position"]


def load_and_engineer() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)

    # Label -- identical definition to the reference (trend_direction == "down"),
    # kept identical on purpose so results are comparable on the same target.
    df["is_declining_label"] = (df["trend_direction"] == "down").astype(int)

    # Fill numeric/categorical blanks the same way the reference prep step does.
    for col in BASE_NUMERIC + LOG_SOURCE:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    for col in CATEGORICAL:
        df[col] = df[col].fillna("unknown").astype(str)

    for col in LOG_SOURCE:
        df[f"log_{col}"] = np.log1p(df[col])

    # --- Novel features: peer-relative z-scores -----------------------------
    # How a page's CTR / engagement / position compares to same content_type +
    # position_tier peers, instead of only looking at absolute numbers.
    for target in PEER_TARGET_COLS:
        grp = df.groupby(PEER_GROUP_COLS)[target]
        mean = grp.transform("mean")
        std = grp.transform("std").replace(0, np.nan)
        df[f"{target}_peer_z"] = ((df[target] - mean) / std).fillna(0)

    # Demand x difficulty interaction (visible page in an easy keyword slot)
    df["visibility_opportunity"] = df["log_impressions_90d"] * (1 - df["competition"].fillna(0))

    return df


def build_feature_lists():
    numeric = BASE_NUMERIC + [f"log_{c}" for c in LOG_SOURCE]
    numeric += [f"{c}_peer_z" for c in PEER_TARGET_COLS]
    numeric += ["visibility_opportunity"]
    return numeric, CATEGORICAL


def make_pipeline(model, numeric_features, categorical_features):
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric_features),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_features),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def precision_at_k(y_true, scores, k=50):
    order = np.argsort(scores)[::-1][:k]
    return float(np.mean(np.asarray(y_true)[order]))


def repeated_grouped_cv(df, numeric_features, categorical_features, n_splits=5, n_repeats=3):
    X = df[numeric_features + categorical_features]
    y = df["is_declining_label"].values
    groups = df["client_id"].values

    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=RANDOM_SEED),
    }

    results = {name: {"roc_auc": [], "avg_precision": [], "precision_at_50": []} for name in models}

    for repeat in range(n_repeats):
        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED + repeat)
        for train_idx, test_idx in skf.split(X, y, groups=groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            for name, model in models.items():
                pipe = make_pipeline(model, numeric_features, categorical_features)
                pipe.fit(X_train, y_train)
                proba = pipe.predict_proba(X_test)[:, 1]
                results[name]["roc_auc"].append(roc_auc_score(y_test, proba))
                results[name]["avg_precision"].append(average_precision_score(y_test, proba))
                results[name]["precision_at_50"].append(precision_at_k(y_test, proba, k=50))

    summary = {}
    for name, metrics in results.items():
        summary[name] = {
            m: {"mean": float(np.mean(v)), "std": float(np.std(v))} for m, v in metrics.items()
        }
    return summary


def fit_final_model_and_score(df, numeric_features, categorical_features):
    """Fit HistGBM on ALL data with a held-out client group for a top-N preview,
    then produce the demand-weighted opportunity_score for every row."""
    X = df[numeric_features + categorical_features]
    y = df["is_declining_label"].values
    groups = df["client_id"].values

    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    train_idx, test_idx = next(skf.split(X, y, groups=groups))

    pipe = make_pipeline(
        HistGradientBoostingClassifier(random_state=RANDOM_SEED), numeric_features, categorical_features
    )
    pipe.fit(X.iloc[train_idx], y[train_idx])

    proba_test = pipe.predict_proba(X.iloc[test_idx])[:, 1]
    test_auc = roc_auc_score(y[test_idx], proba_test)
    test_precision_50 = precision_at_k(y[test_idx], proba_test, k=50)

    # Refit on everything for the actual scoring queue used in recommendations
    pipe_full = make_pipeline(
        HistGradientBoostingClassifier(random_state=RANDOM_SEED), numeric_features, categorical_features
    )
    pipe_full.fit(X, y)
    proba_all = pipe_full.predict_proba(X)[:, 1]

    out = df[["content_id", "client_id", "content_type", "position_tier",
              "impressions_90d", "avg_position", "trend_direction"]].copy()
    out["decline_probability"] = proba_all
    out["opportunity_score"] = proba_all * df["log_impressions_90d"]
    out = out.sort_values("opportunity_score", ascending=False).reset_index(drop=True)

    return out, {"held_out_test_auc": test_auc, "held_out_test_precision_at_50": test_precision_50}


def main():
    df = load_and_engineer()
    numeric_features, categorical_features = build_feature_lists()

    cv_summary = repeated_grouped_cv(df, numeric_features, categorical_features)
    queue, holdout_metrics = fit_final_model_and_score(df, numeric_features, categorical_features)

    (OUT_DIR / "cv_summary.json").write_text(json.dumps(cv_summary, indent=2, sort_keys=True))
    (OUT_DIR / "holdout_metrics.json").write_text(json.dumps(holdout_metrics, indent=2))
    queue.head(200).to_csv(OUT_DIR / "opportunity_queue_top200.csv", index=False)

    print(json.dumps({"cv_summary": cv_summary, "holdout_metrics": holdout_metrics}, indent=2))


if __name__ == "__main__":
    main()
