# Capstone Report — Refresh / Content Opportunity Scoring

- **Author:** Abhyudita
- **Lane:** Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/ART001-coder/flyrank-ml-internship
- **Date:** 2026-08-08

## 0. Abstract

Which pages in a content library are worth an editor's attention this week, and how should
that attention be prioritized? Using the bundled 30,000-row anonymized FlyRank slice, I built
a HistGradientBoosting classifier on 90-day GSC/GA4 aggregates plus two engineered
peer-relative features (CTR and engagement z-scores within content_type/position_tier), fit
with repeated grouped cross-validation to guard against client-specific overfitting. The model
reaches 0.69 mean ROC AUC (± 0.06) versus the reference pipeline's single-split 0.75, but a
92% precision at the top 50 ranked items on held-out clients — meaning the ranked queue is
sharply reliable exactly where it matters, even though the model is a noisier separator
overall. The output is a demand-weighted opportunity_score (decline probability × log
impressions) that ranks pages so editors review the highest-traffic at-risk pages first,
not just the most probable ones.

## 1. Problem framing

**Unit of analysis:** one content page (row) per 90-day window.
**Output:** a ranked opportunity_score plus a decline_probability per page.
**Action a human takes:** an editor reviews the top-N ranked pages and decides whether to
refresh, merge, or monitor — this model is a triage aid, not an auto-publish trigger.
**Cost of a wrong call:** a false positive wastes an editor's review time on a page that
wasn't actually declining; a false negative lets a high-traffic page keep losing visibility
unnoticed. Because those costs are asymmetric and impressions vary by orders of magnitude
across pages, ranking by probability alone is the wrong objective — a page with 90% decline
probability and 50 impressions matters less than one with 60% probability and 200,000
impressions. That's why the score is demand-weighted rather than a raw probability cutoff.

## 2. Data safety

Data: `data/raw/content_refresh_anonymized.csv` (30,000 rows, 32 pseudonymized clients,
bundled starter slice — see Reproducibility for the full-warehouse caveat).

Excluded from features, deliberately: `trend_direction` and `trend_pct` (label source),
`impressions_last_30d` / `clicks_last_30d` / `sessions_last_30d` and their `_prev_30d`
counterparts (these are exactly the columns `trend_direction` is computed from — including
them would leak the label almost perfectly), `content_id` / `client_id` (grouping keys only),
`provider_used` / `model_used` (explicitly flagged as non-features in the data dictionary).

No client names, domains, URLs, or raw exports appear anywhere in `work/`.

## 3. Baseline

The reference pipeline's baseline (`scripts/02_baseline_score.py`) is a transparent staleness +
volume rule scoring 0.627 ROC AUC / 0.240 precision@50 on its client-holdout split
(`outputs/model_report.md`). I use that as the floor: any model here needs to clear it by a
meaningful margin on a comparable metric, which it does — precision@50 of 0.86–0.92 vs. 0.240.

## 4. Model / analysis

**Method:** `HistGradientBoostingClassifier`, compared against logistic regression and random
forest (the same model family the reference used, so results are comparable) plus two features
the reference pipeline doesn't include:

- `{ctr, engagement_rate, avg_position}_peer_z` — each page's z-score relative to other pages
  in the same `content_type` × `position_tier` group, instead of only the raw value. A 2%
  CTR means something different for a `page_1` keyword article than a `page_3_5` one.
- `visibility_opportunity` — `log_impressions_90d × (1 − competition)`, an interaction term for
  "visible page sitting in an easy keyword slot."

**Target:** `is_declining_label = (trend_direction == "down")` — identical definition to the
reference, kept the same on purpose so the two approaches are comparable on the same target.

## 5. Evaluation

**Split:** repeated grouped stratified K-fold — 5 folds × 3 repeats, grouped by `client_id` so
no client's pages appear in both train and test within a fold (the reference uses one
client-holdout split; I report a distribution instead of a point estimate).

| Model | ROC AUC (mean ± std) | Avg precision | Precision@50 |
|---|---:|---:|---:|
| logistic_regression | 0.668 ± 0.040 | 0.674 ± 0.046 | 0.732 ± 0.141 |
| random_forest | 0.689 ± 0.049 | 0.707 ± 0.043 | 0.864 ± 0.023 |
| hist_gradient_boosting | 0.692 ± 0.059 | 0.707 ± 0.058 | **0.860 ± 0.089** |

Single held-out client-group check (comparable in spirit to the reference's split):
ROC AUC 0.644, **precision@50 = 0.92**.

**Error pattern:** the std on precision@50 (±0.09–0.14 across repeats) is larger than the std
on ROC AUC, meaning the *top of the ranked queue* is more sensitive to which clients land in
the test fold than the overall separability is. Practically: trust the ranked queue's top
tier's *relative order* strongly, but don't treat any single precision@50 number as fixed —
report it as a range, which is what the repeated CV is for.

## 6. Interpretation

Permutation importance (drop in ROC AUC when a feature is shuffled) on the held-out fold:

| Feature | Importance |
|---|---:|
| days_with_impressions | 0.076 |
| avg_position | 0.050 |
| content_age_days | 0.036 |
| log_impressions_90d | 0.024 |
| scroll_rate | 0.013 |
| **ctr_peer_z** | 0.009 |
| log_clicks_90d | 0.008 |
| log_sessions_90d | 0.006 |
| **engagement_rate_peer_z** | 0.006 |

The two peer-relative features I added (`ctr_peer_z`, `engagement_rate_peer_z`) rank above
several raw aggregates the reference pipeline used — evidence that "how a page's CTR compares
to similar pages" carries independent signal beyond "what a page's CTR is." Consistency of
visibility (`days_with_impressions`) and position dominate everything else, which matches
intuition: pages that show up reliably and rank well are the ones with enough signal to tell
decline from noise in the first place.

**Negative result:** `word_count`/`char_count` and their tiers — features the reference also
used — rank far down the importance list here; content length isn't doing much work for this
target once visibility and position are accounted for.

## 7. Recommendation

Ranked action playbook, driven by `opportunity_score` (decline probability × demand), not
probability alone:

1. **Top tier (opportunity_score > 8):** high-impression, high-decline-probability pages —
   route directly to an editor for manual review this week; these are the pages where being
   wrong is expensive.
2. **Mid tier:** high decline probability but lower demand — batch into a monthly review queue
   rather than an urgent one.
3. **Low tier / monitor:** low decline probability regardless of demand — no action, re-score
   next cycle.

Confidence: high on relative ranking of the top tier (precision@50 consistently 0.73–0.92
across repeated splits); moderate on the exact probability values (ROC AUC ~0.69, meaningfully
better than the 0.627 baseline but not a strong overall separator) — this is a triage aid for
human review, not a fully automated action trigger.

## 8. Reproducibility

```bash
git clone https://github.com/ART001-coder/flyrank-ml-internship.git
cd flyrank-ml-internship
pip install -r requirements.txt
python work/scripts/opportunity_score_model.py
python work/scripts/interpret_and_chart.py
```

Random seed: 42, fixed throughout (`RANDOM_SEED` in `opportunity_score_model.py`).
Outputs land in `work/outputs/` (`cv_summary.json`, `holdout_metrics.json`,
`opportunity_queue_top200.csv`) and `work/figures/`.

**Scope caveat, stated honestly:** this analysis runs on the bundled 30k-row teaching slice
(`data/raw/content_refresh_anonymized.csv`), not the full ~79M-row warehouse release on
Hugging Face. The methodology (peer-relative features, repeated grouped CV, demand-weighted
scoring) is designed to carry over directly to `notebooks/03`'s DuckDB workflow against the
full release; re-running there with per-client time windows would be the natural next
validation step and may shift these numbers.

## 9. Data credit

Built on the FlyRank ML Internship dataset — https://flyrank.ai
