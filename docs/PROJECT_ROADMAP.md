# AutoWorth-AI Roadmap

## Milestone 1 — Project Setup
- Repository created
- Python virtual environment configured
- ML project structure created
- Dataset obtained

Status: COMPLETE

## Milestone 2 — Data Understanding
- Loaded Cardekho dataset
- 15,411 records
- 14 original columns
- Checked data types
- Checked missing values
- Checked duplicates
- Identified `selling_price` as target

Status: COMPLETE

## Milestone 3 — Exploratory Data Analysis
- Vehicle age distribution
- Selling price distribution
- Brand distribution
- KM driven distribution
- Vehicle age vs selling price
- KM driven vs selling price
- Selling price outlier investigation
- Correlation analysis
- Brand/fuel/transmission analysis
- Investigated `car_name`, `brand`, and `model`

Key findings:
- `selling_price` is highly right-skewed
- `max_power` has the strongest numerical correlation with price (~0.75)
- `engine` correlation with price is ~0.59
- `vehicle_age` correlation is ~-0.24
- `km_driven` correlation is weak (~-0.08)
- `engine` and `max_power` are strongly correlated (~0.81)
- 86 records have `km_driven > 200,000`
- Maximum `km_driven` is 3,800,000
- `car_name` is redundant with `brand` + `model`

Feature decisions:
- Remove `Unnamed: 0`
- Remove `car_name`
- Keep `brand`
- Keep `model`
- Keep remaining features for now

Status: COMPLETE

## Milestone 4 — Data Cleaning

Tasks:
- Investigate extreme `km_driven` values
- Establish a defensible outlier/data-quality strategy
- Clean invalid records
- Preserve raw dataset
- Create processed dataset

Status: COMPLETE

## Milestone 5 — Feature Engineering
- Encode categorical variables
- Prepare numerical features
- Investigate useful transformations
- Build reproducible preprocessing pipeline

Status: COMPLETE

## Milestone 6 — Model Training
- Train baseline model
- Evaluate using MAE, RMSE, R²
- Compare multiple regression algorithms

## Milestone 7 — Model Improvement
- Feature engineering experiments
- Hyperparameter tuning
- Cross-validation
- Select final model

## Milestone 8 — Prediction Backend
- Build prediction service/API
- Load trained model
- Validate input
- Return predicted used-car price

## Milestone 9 — Frontend
- Build AutoWorth-AI interface
- Car details form
- Price prediction
- Results visualization

## Milestone 10 — Deployment
- Deploy backend
- Deploy frontend
- Document architecture and setup