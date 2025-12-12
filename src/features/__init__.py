"""
Feature engineering module for IRP platform.

Provides feature extraction for prediction market timeseries.
"""

from src.features.feature_engineering import (
    FeatureEngineer,
    FeatureExtractor,
    ReturnFeatures,
    VolatilityFeatures,
    LiquidityFeatures,
    SpreadFeatures,
    DepthFeatures,
    RegimeFeatures,
    engineer_features,
)

__all__ = [
    "FeatureEngineer",
    "FeatureExtractor",
    "ReturnFeatures",
    "VolatilityFeatures",
    "LiquidityFeatures",
    "SpreadFeatures",
    "DepthFeatures",
    "RegimeFeatures",
    "engineer_features",
]
