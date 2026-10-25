# Milestone 5: Feature Engineering and Preprocessing

## Input and target

The preprocessing workflow reads `ml/data/processed/cardekho_model_ready.csv`, the 15,235-row Milestone 4 output. `selling_price` is separated before splitting and is not used by the preprocessing transformer.

Before splitting or encoding, the feature table applies the deterministic mapping `ISUZU -> Isuzu`. This merges the two observed `ISUZU` rows and seven `Isuzu` rows into nine `Isuzu` rows. It is an input-label correction only and does not use the target. The source processed CSV remains unchanged.

## Leakage-safe split

`train_test_split` creates an 80/20 holdout split with `random_state=42`: 12,188 training rows and 3,047 test rows. The split occurs before fitting imputers, scalers, or encoders. Only the training features are passed to `preprocessor.fit_transform`; test features are passed only to `preprocessor.transform`.

## Feature handling

Numerical features are `vehicle_age`, `km_driven`, `mileage`, `engine`, `max_power`, and `seats`. They use median imputation and standard scaling. Categorical features are `brand`, `model`, `seller_type`, `fuel_type`, and `transmission_type`. They use most-frequent imputation and one-hot encoding with `handle_unknown="ignore"`.

The resulting feature matrix is expected to be sparse because each categorical feature contributes one active encoded value per record. With the fixed training split, it has 30 brand categories, 118 model categories, and 158 one-hot features in total; together with six numerical features, the transformed matrix has 164 columns. The notebook inspects train/test dimensions, feature names, sparsity, and density. It also verifies that an unseen brand can be transformed without changing the learned feature layout.

The reusable `ml.preprocessing.build_preprocessor` factory embeds the same deterministic brand normalization before its learned column transformer. The Milestone 5 notebook saves its training-partition artifact as `ml/models/preprocessor_random_split.joblib`; the full-data `ml/models/preprocessor.joblib` is produced by the canonical Milestone 6 run and should be used for prediction integration.

## Scope boundary

No model is trained, evaluated, selected, or saved in this milestone. The target is not transformed. Those decisions belong to later modelling milestones.
