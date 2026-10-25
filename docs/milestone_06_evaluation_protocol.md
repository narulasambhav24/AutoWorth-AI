# Milestone 6: Evaluation Protocol for Repeated Feature Profiles

## Risk identified during Milestone 5

After Milestone 4 removes exact duplicate modelling rows, the canonical feature table still contains 953 distinct repeated full feature profiles (2,202 rows participate in those repeated groups). Every repeated profile has more than one `selling_price`, so this is not duplicate-target leakage. It is an evaluation-design risk: identical vehicle profiles can land on both sides of an ordinary random split.

With the fixed Milestone 5 random split (`test_size=0.20`, `random_state=42`), 390 of 3,047 test rows (12.8%) have an identical canonical feature profile in the training set. This does not invalidate preprocessing: the target is never used by the transformer and preprocessing is fitted only on training features. It can, however, make a model's holdout score somewhat optimistic because those profile patterns have already been observed during training.

## Milestone 6 comparison

Milestone 6 must train and score each candidate model under both of these protocols, using fresh preprocessing fitted within that protocol's training data:

1. **Ordinary random split** — retain the current 80/20 split as a familiar baseline and to quantify the effect of profile overlap.
2. **Grouped split by full canonical feature profile** — use `GroupShuffleSplit(test_size=0.20, random_state=42)` with group labels formed from all 11 non-target features after the deterministic `ISUZU -> Isuzu` normalization. This prevents any identical feature profile from occurring in both train and test. The verified split has 12,175 training rows, 3,060 test rows, and zero shared profile groups.

Report MAE, RMSE, and R² for both protocols. Do not reuse a fitted preprocessor across protocols; each training partition must fit its own imputer, scaler, and encoder.

## Chosen primary evaluation

Use the grouped split as the primary Milestone 6 model-selection result, because it measures generalization to feature profiles not already present in the training partition. Keep the ordinary random-split result as a secondary comparator: the gap quantifies the potential optimism caused by profile overlap and preserves continuity with the initial holdout design. This decision changes the evaluation design only; it does not alter the Milestone 5 preprocessing pipeline.

## Canonical implementation

Milestone 6 is implemented in `ml/notebooks/06_canonical_baseline_evaluation.ipynb`. It uses `ml.preprocessing.build_preprocessor` for every evaluation and for the final saved Random Forest baseline. That reusable pipeline embeds the deterministic `ISUZU -> Isuzu` transformation before its learned column transformer, so the same rule runs at fit time and prediction time.

The notebook fits a new preprocessing instance for every model and split, compares Dummy, Linear, Ridge, and Random Forest regressors using MAE, RMSE, and R², then fits the selected Random Forest on all cleaned rows. The resulting `ml/models/autoworth_random_forest_baseline.joblib` and `ml/models/preprocessor.joblib` contain `brand_Isuzu` but not `brand_ISUZU`.
