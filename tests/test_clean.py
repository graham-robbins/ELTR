"""
Unit tests for cleaning module.

Run tests with: pytest tests/test_clean.py -v
After installing package with: pip install -e .
"""

import numpy as np
import pandas as pd
import pytest

from src.utils.config import load_config
from src.utils.types import ContractTimeseries, MarketDataset
from src.clean.cleaner import (
    DataCleaner,
    Imputer,
    Winsorizer,
    PriceValidator,
    SpreadValidator,
    TimestampValidator,
    ImputationMethod,
    ValidationResult,
)


class TestImputer:
    """Tests for Imputer class."""

    def test_forward_fill(self):
        """Test forward fill imputation."""
        imputer = Imputer(method=ImputationMethod.FORWARD_FILL)
        df = pd.DataFrame({
            "price": [50.0, np.nan, np.nan, 55.0],
        })

        result = imputer.impute(df, columns=["price"])

        assert result["price"].isna().sum() == 0
        assert result["price"].iloc[1] == 50.0
        assert result["price"].iloc[2] == 50.0

    def test_linear_interpolation(self):
        """Test linear interpolation."""
        imputer = Imputer(method=ImputationMethod.LINEAR)
        df = pd.DataFrame({
            "price": [50.0, np.nan, 60.0],
        })

        result = imputer.impute(df, columns=["price"])

        assert result["price"].iloc[1] == 55.0

    def test_bid_ask_imputation(self):
        """Test specialized bid-ask imputation."""
        imputer = Imputer()
        df = pd.DataFrame({
            "yes_bid_c": [49.0, np.nan, 51.0],
            "yes_ask_c": [51.0, np.nan, 53.0],
        })

        result = imputer.impute_bid_ask(df)

        assert result["yes_bid_c"].iloc[1] == 49.0
        assert result["yes_ask_c"].iloc[1] == 51.0


class TestWinsorizer:
    """Tests for Winsorizer class."""

    def test_winsorize_outliers(self):
        """Test outlier winsorization."""
        winsorizer = Winsorizer(limits=(0.1, 0.1))
        df = pd.DataFrame({
            "returns": [0.01, 0.02, -0.5, 0.03, 0.5, 0.02],
        })

        result, n_modified = winsorizer.winsorize(df, columns=["returns"])

        assert n_modified > 0
        assert result["returns"].min() > -0.5
        assert result["returns"].max() < 0.5

    def test_winsorize_preserves_normal_values(self):
        """Test that most values are preserved and only tails are modified."""
        winsorizer = Winsorizer(limits=(0.01, 0.01))
        np.random.seed(42)  # Deterministic for reproducibility
        df = pd.DataFrame({
            "returns": np.random.normal(0, 0.01, 100).tolist(),
        })

        result, n_modified = winsorizer.winsorize(df, columns=["returns"])

        # With 1% limits on each tail, at most ~2 values should be modified
        assert n_modified <= 4, f"Too many values modified: {n_modified}"

        # Values not at the extremes should be unchanged
        original_sorted = df["returns"].sort_values()
        result_sorted = result["returns"].sort_values()

        # Middle 96% of values should be identical (excluding ~2 on each end)
        middle_original = original_sorted.iloc[2:-2].values
        middle_result = result_sorted.iloc[2:-2].values
        np.testing.assert_array_almost_equal(middle_original, middle_result, decimal=10)


class TestPriceValidator:
    """Tests for PriceValidator class."""

    def test_valid_prices(self):
        """Test validation passes for valid prices."""
        validator = PriceValidator(min_price=0, max_price=100)
        df = pd.DataFrame({
            "price_c": [25.0, 50.0, 75.0],
        })

        result = validator.validate(df)

        assert result.is_valid
        assert len(result.errors) == 0

    def test_invalid_prices(self):
        """Test validation fails for invalid prices."""
        validator = PriceValidator(min_price=0, max_price=100)
        df = pd.DataFrame({
            "price_c": [-5.0, 50.0, 150.0],
        })

        result = validator.validate(df)

        assert not result.is_valid
        assert len(result.errors) == 2


class TestSpreadValidator:
    """Tests for SpreadValidator class."""

    def test_valid_spreads(self):
        """Test validation passes for normal spreads."""
        validator = SpreadValidator(max_spread_pct=0.5)
        df = pd.DataFrame({
            "yes_bid_c": [48.0, 49.0, 50.0],
            "yes_ask_c": [52.0, 51.0, 52.0],
        })

        result = validator.validate(df)

        assert result.is_valid

    def test_wide_spreads(self):
        """Test validation fails for wide spreads."""
        validator = SpreadValidator(max_spread_pct=0.1)
        df = pd.DataFrame({
            "yes_bid_c": [20.0],
            "yes_ask_c": [80.0],
        })

        result = validator.validate(df)

        assert not result.is_valid


class TestTimestampValidator:
    """Tests for TimestampValidator class."""

    def test_valid_timestamps(self):
        """Test validation passes for valid timestamps."""
        validator = TimestampValidator()
        df = pd.DataFrame({
            "price_c": [50.0, 51.0, 52.0],
        }, index=pd.date_range("2024-01-01", periods=3, freq="min", tz="UTC"))

        result = validator.validate(df)

        assert result.is_valid

    def test_non_datetime_index(self):
        """Test validation fails for non-datetime index."""
        validator = TimestampValidator()
        df = pd.DataFrame({
            "price_c": [50.0, 51.0, 52.0],
        })

        result = validator.validate(df)

        assert not result.is_valid


class TestDataCleaner:
    """Tests for DataCleaner class."""

    def test_clean_contract(self):
        """Test cleaning a single contract."""
        config = load_config()
        cleaner = DataCleaner(config)

        df = pd.DataFrame({
            "price_c": [50.0, np.nan, 52.0, 53.0, 54.0],
            "volume": [100, 150, 200, 250, 300],
            "yes_bid_c": [49.0, 49.0, 51.0, 52.0, 53.0],
            "yes_ask_c": [51.0, 51.0, 53.0, 54.0, 55.0],
        }, index=pd.date_range("2024-01-01", periods=5, freq="min", tz="UTC"))

        contract = ContractTimeseries(
            contract_id="TEST",
            category="Sports",
            data=df,
        )

        cleaned, stats = cleaner.clean_contract(contract)

        assert cleaned.n_observations > 0
        assert stats.original_rows == 5
        assert stats.nulls_imputed >= 0

    def test_clean_removes_insufficient_obs(self):
        """Test cleaning removes contracts with too few observations."""
        config = load_config()
        config.cleaning.min_observations = 100  # High threshold
        cleaner = DataCleaner(config)

        df = pd.DataFrame({
            "price_c": [50.0, 51.0, 52.0],
        }, index=pd.date_range("2024-01-01", periods=3, freq="min", tz="UTC"))

        contract = ContractTimeseries(
            contract_id="TEST",
            category="Sports",
            data=df,
        )

        dataset = MarketDataset()
        dataset.add(contract)

        cleaned, stats = cleaner.clean_dataset(dataset)

        assert len(cleaned) == 0


class TestValidationResult:
    """Tests for ValidationResult class."""

    def test_valid_result(self):
        """Test valid result boolean conversion."""
        result = ValidationResult(is_valid=True)

        assert bool(result)
        assert result.is_valid

    def test_invalid_result(self):
        """Test invalid result with errors."""
        result = ValidationResult(is_valid=False, errors=["Error 1", "Error 2"])

        assert not bool(result)
        assert len(result.errors) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
