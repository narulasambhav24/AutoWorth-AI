# Milestone 4: `km_driven` data-quality decision

## Scope

This decision is based on `ml/data/raw/cardekho_dataset.csv` (15,411 rows) and does not alter that file. The modelling target is `selling_price`.

## Audit results

| Range | Records | Share |
| --- | ---: | ---: |
| Above 200,000 km | 86 | 0.56% |
| Above 500,000 km | 12 | 0.08% |
| Above 1,000,000 km | 2 | 0.01% |

The 200k–500k tail is mostly older diesel Toyota Innova, Mahindra Bolero/Scorpio, and similar utility vehicles. These values have mileage rates compatible with intensive use, so they are retained. A simple percentile rule would incorrectly discard valid high-mileage cars.

The 12 records above 500k km are reviewed individually. Five have an exact equality between `km_driven` and the target `selling_price` (525k, 675k, 825k, 850k, and 950k). This equality occurs in 50% of records from 500k–1m, versus roughly 0.10% below 200k, which is strong evidence of a source-field copy error. Two further records imply more than 250,000 km/year: a two-year Toyota Fortuner at 590k km and a five-year Mahindra XUV500 at 3.8m km. That threshold is about 685 km every day for the full ownership period; these two values are not credible as consumer-vehicle odometer readings.

The 1.325m-km, nine-year Mercedes-Benz E-Class is unusual (about 147k km/year) but is not demonstrably erroneous from this dataset. It is retained, as are the other non-flagged values above 500k.

## Cleaning rule

For training data only, remove a row if either condition holds:

1. `km_driven > 500_000` and `km_driven == selling_price`; or
2. `km_driven / vehicle_age > 250_000` where `vehicle_age > 0`.

This removes seven rows (0.045% of the data). The `km_driven == selling_price` check is a one-time label-quality audit only; it is not a prediction feature and is not retained in the modelling table, so prediction-time leakage is avoided. Odometer values above 200k and most above 500k are retained. The audit manifest records the seven raw row IDs and their specific exclusion reasons.

## Other agreed column decisions

`Unnamed: 0` is an exported row index and `car_name` is removed. That removal exposes 167 exact duplicate modelling rows, so the cleaning process keeps the first occurrence and removes the remaining 167. `brand` and `model` are retained. The two rows with `seats == 0` are removed because zero is not a valid seat count. The five rows with `vehicle_age == 0` are retained as potentially same-year vehicles; they are intentionally excluded from the annual-km calculation, and the audit asserts that no age is negative. The heavily right-skewed target and legitimate luxury-car prices are not clipped or removed at this cleaning stage; target transformation and robust evaluation belong in a later modelling milestone.

## Reproducibility

Run `ml/notebooks/03_data_cleaning.ipynb` from its own directory. It reads the raw CSV and writes the derived, model-ready table to `ml/data/processed/cardekho_model_ready.csv` and the complete exclusion manifest to `ml/data/processed/cleaning_exclusions_manifest.csv`. The raw data is read-only. The final model-ready table has 15,235 rows and 12 columns.
