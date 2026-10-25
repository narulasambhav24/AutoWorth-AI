"""Factory for the leakage-safe AutoWorth-AI preprocessing pipeline."""

from collections.abc import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .brand_normalization import normalize_brand_labels


def build_preprocessor(
    numerical_features: Sequence[str], categorical_features: Sequence[str]
) -> Pipeline:
    """Build an unfitted preprocessing pipeline for one training partition.

    The first step normalizes brand labels deterministically. All learned operations
    are fitted only when the returned pipeline is fitted by the caller.
    """
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    columns = ColumnTransformer(
        transformers=[
            ("numeric", numerical_pipeline, list(numerical_features)),
            ("categorical", categorical_pipeline, list(categorical_features)),
        ],
        remainder="drop",
        sparse_threshold=0.30,
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[
            (
                "brand_normalizer",
                FunctionTransformer(
                    normalize_brand_labels,
                    validate=False,
                    feature_names_out="one-to-one",
                ),
            ),
            ("columns", columns),
        ]
    )
