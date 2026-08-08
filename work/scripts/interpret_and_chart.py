from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedGroupKFold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from opportunity_score_model import (
    ROOT, OUT_DIR, RANDOM_SEED, load_and_engineer, build_feature_lists, make_pipeline,
)

FIG_DIR = ROOT / "work" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = load_and_engineer()
    numeric_features, categorical_features = build_feature_lists()
    X = df[numeric_features + categorical_features]
    y = df["is_declining_label"].values
    groups = df["client_id"].values

    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    train_idx, test_idx = next(skf.split(X, y, groups=groups))

    pipe = make_pipeline(HistGradientBoostingClassifier(random_state=RANDOM_SEED), numeric_features, categorical_features)
    pipe.fit(X.iloc[train_idx], y[train_idx])

    perm = permutation_importance(
        pipe, X.iloc[test_idx], y[test_idx], n_repeats=5, random_state=RANDOM_SEED, scoring="roc_auc", n_jobs=-1
    )
    feat_names = numeric_features + categorical_features
    importances = pd.Series(perm.importances_mean, index=feat_names).sort_values(ascending=False)
    importances.head(12).to_json(OUT_DIR / "permutation_importance_top12.json", indent=2)

    # Chart 1: top permutation importances
    top = importances.head(10).iloc[::-1]
    plt.figure(figsize=(7, 5))
    plt.barh(top.index, top.values, color="#2563eb")
    plt.xlabel("Permutation importance (drop in ROC AUC)")
    plt.title("Top features — Opportunity Scoring model")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "top_feature_importance.png", dpi=150)
    plt.close()

    # Chart 2: CV metric comparison across models
    cv_summary = json.loads((OUT_DIR / "cv_summary.json").read_text())
    models = list(cv_summary.keys())
    means = [cv_summary[m]["roc_auc"]["mean"] for m in models]
    stds = [cv_summary[m]["roc_auc"]["std"] for m in models]
    plt.figure(figsize=(6, 4))
    plt.bar(models, means, yerr=stds, capsize=5, color=["#94a3b8", "#60a5fa", "#2563eb"])
    plt.ylabel("ROC AUC (mean ± std, 5-fold x 3 repeats)")
    plt.title("Model comparison — repeated grouped CV")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "model_comparison_cv.png", dpi=150)
    plt.close()

    print(importances.head(12))


if __name__ == "__main__":
    main()
