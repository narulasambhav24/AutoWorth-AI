"""Deterministic feature-label normalization used at fit and prediction time."""

import pandas as pd


BRAND_NORMALIZATION = {"ISUZU": "Isuzu"}


def normalize_brand_labels(features: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *features* with known brand spelling variants canonicalized.

    This is deliberately target-independent and does not learn from the data. Keeping
    it in an importable module makes joblib artifacts portable across notebooks and
    prediction services.
    """
    if not isinstance(features, pd.DataFrame):
        raise TypeError("Brand normalization requires a pandas DataFrame with named columns.")
    if "brand" not in features.columns:
        raise ValueError("Expected a 'brand' feature column for canonicalization.")

    normalized = features.copy()
    normalized["brand"] = normalized["brand"].replace(BRAND_NORMALIZATION)
    return normalized
