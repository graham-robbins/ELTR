"""
Edge case tests for the ELTR platform.

Tests unusual inputs, boundary conditions, and error handling.

Run tests with: pytest tests/test_edge_cases.py -v
After installing package with: pip install -e .
"""

import numpy as np
import pandas as pd
import pytest

from src.utils.config import load_config, CleaningConfig, RollingWindowConfig
from src.utils.types import ContractTimeseries, MarketDataset
from src.clean.cleaner import DataCleaner, Imputer, Winsorizer, PriceValidator
from src.features.feature_engineering import (
    ReturnFeatures,
    VolatilityFeatures,
    LiquidityFeatures,
    SpreadFeatures,
)
from src.ingest.kalshi_loader import sanitize_contract_id, sanitize_filename


class TestSecurityEdgeCases:
    """Tests for security-related edge cases."""

    def test_path_traversal_contract_id(self):
        """Test that path traversal attempts are blocked."""
        with pytest.raises(ValueError, match="Invalid contract ID"):
            sanitize_contract_id("../../../etc/passwd")

    def test_path_traversal_with_backslash(self):
        """Test path traversal with backslashes."""
        with pytest.raises(ValueError, match="Invalid contract ID"):
            sanitize_contract_id("..\\..\\etc\\passwd")

    def test_empty_contract_id(self):
        """Test that empty contract ID is rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            sanitize_contract_id("")

    def test_valid_contract_id(self):
        """Test that valid contract IDs pass through."""
        assert sanitize_contract_id("VALID_CONTRACT_123") == "VALID_CONTRACT_123"
        assert sanitize_contract_id("test-contract") == "test-contract"

    def test_path_traversal_filename(self):
        """Test that path traversal in filenames is blocked."""
        with pytest.raises(ValueError):
            sanitize_filename("../../malicious.csv")

    def test_empty_filename(self):
        """Test that empty filename is rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            sanitize_filename("")


class TestEmptyDataFrames:
    """Tests for empty DataFrame handling."""

    def test_return_features_empty_df(self, empty_df):
        """Test ReturnFeatures on empty DataFrame."""
        extractor = ReturnFeatures()
        result = extractor.extract(empty_df)
        assert len(result) == 0

    def test_volatility_features_empty_df(self, empty_df):
        """Test VolatilityFeatures on empty DataFrame."""
        extractor = VolatilityFeatures()
        result = extractor.extract(empty_df)
        assert len(result) == 0

    def test_spread_features_empty_df(self, empty_df):
        """Test SpreadFeatures on empty DataFrame."""
        extractor = SpreadFeatures()
        result = extractor.extract(empty_df)
        assert len(result) == 0

    def test_cleaner_empty_dataset(self, config):
        """Test DataCleaner on empty dataset."""
        dataset = MarketDataset()
        cleaner = DataCleaner(config)
        cleaned, stats = cleaner.clean_dataset(dataset)
        assert len(cleaned) == 0
        assert len(stats) == 0


class TestSingleRowDataFrames:
    """Tests for single-row DataFrame handling."""

    def test_return_features_single_row(self, single_row_df):
        """Test ReturnFeatures on single-row DataFrame."""
        extractor = ReturnFeatures()
        result = extractor.extract(single_row_df)
        # Should still produce the DataFrame, returns will be NaN
        assert len(result) == 1
        assert pd.isna(result["pct_return"].iloc[0])

    def test_volatility_features_single_row(self, single_row_df):
        """Test VolatilityFeatures on single-row DataFrame."""
        # Add return column first
        single_row_df["pct_return"] = np.nan
        extractor = VolatilityFeatures()
        result = extractor.extract(single_row_df)
        assert len(result) == 1

    def test_spread_features_single_row(self, single_row_df):
        """Test SpreadFeatures on single-row DataFrame."""
        extractor = SpreadFeatures()
        result = extractor.extract(single_row_df)
        assert len(result) == 1
        assert "spread" in result.columns


class TestNaNHandling:
    """Tests for NaN value handling."""

    def test_imputer_all_nan_column(self):
        """Test Imputer when entire column is NaN."""
        df = pd.DataFrame({
            "price": [np.nan, np.nan, np.nan],
        })
        imputer = Imputer()
        result = imputer.impute(df, columns=["price"])
        # All values remain NaN (no valid values to propagate)
        assert result["price"].isna().all()

    def test_winsorizer_all_nan_column(self):
        """Test Winsorizer when entire column is NaN."""
        df = pd.DataFrame({
            "returns": [np.nan, np.nan, np.nan],
        })
        winsorizer = Winsorizer()
        result, n_modified = winsorizer.winsorize(df, columns=["returns"])
        assert n_modified == 0

    def test_price_validator_all_nan(self):
        """Test PriceValidator when all prices are NaN."""
        df = pd.DataFrame({
            "price_c": [np.nan, np.nan, np.nan],
        })
        validator = PriceValidator()
        result = validator.validate(df)
        # Should be valid (empty series has no violations)
        assert result.is_valid


class TestExtremeValues:
    """Tests for extreme value handling."""

    def test_price_at_boundaries(self, config):
        """Test prices exactly at boundaries."""
        df = pd.DataFrame({
            "price_c": [0.0, 100.0, 50.0],
            "volume": [100, 100, 100],
            "yes_bid_c": [0.0, 99.0, 49.0],
            "yes_ask_c": [1.0, 100.0, 51.0],
        }, index=pd.date_range("2024-01-01", periods=3, freq="min", tz="UTC"))

        contract = ContractTimeseries(
            contract_id="BOUNDARY",
            category="Sports",
            data=df,
        )

        cleaner = DataCleaner(config)
        cleaned, stats = cleaner.clean_contract(contract)

        # Boundary values should be clipped
        assert cleaned.data["price_c"].min() >= config.cleaning.min_price
        assert cleaned.data["price_c"].max() <= config.cleaning.max_price

    def test_zero_volume(self, config):
        """Test handling of zero volume."""
        df = pd.DataFrame({
            "price_c": [50.0, 51.0, 52.0],
            "volume": [0, 0, 100],
            "yes_bid_c": [49.0, 50.0, 51.0],
            "yes_ask_c": [51.0, 52.0, 53.0],
        }, index=pd.date_range("2024-01-01", periods=3, freq="min", tz="UTC"))

        extractor = LiquidityFeatures()
        result = extractor.extract(df)

        # volume_surge should handle zero volume
        assert "volume_surge" in result.columns
        # First values with zero volume_ma should be NaN or handled
        assert not np.isinf(result["volume_surge"]).any()

    def test_very_large_values(self):
        """Test handling of very large values."""
        df = pd.DataFrame({
            "price_c": [50.0, 50.0, 50.0],
            "volume": [1, 1e12, 1],  # Very large volume
        }, index=pd.date_range("2024-01-01", periods=3, freq="min", tz="UTC"))

        extractor = LiquidityFeatures()
        result = extractor.extract(df)

        # Should not overflow
        assert not np.isinf(result["volume_ma"]).any()
        assert not np.isinf(result["volume_zscore"]).any()


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_negative_min_observations(self):
        """Test that negative min_observations is rejected."""
        with pytest.raises(ValueError, match="must be non-negative"):
            CleaningConfig(min_observations=-1)

    def test_invalid_winsorize_limits(self):
        """Test that invalid winsorize limits are rejected."""
        with pytest.raises(ValueError, match="must be between 0 and 0.5"):
            CleaningConfig(winsorize_limits=(0.6, 0.1))

    def test_invalid_price_range(self):
        """Test that min_price > max_price is rejected."""
        with pytest.raises(ValueError, match="must be greater than"):
            CleaningConfig(min_price=100, max_price=0)

    def test_invalid_window_order(self):
        """Test that windows not in order are rejected."""
        with pytest.raises(ValueError, match="must satisfy short <= medium <= long"):
            RollingWindowConfig(short=20, medium=10, long=60)

    def test_negative_window(self):
        """Test that negative window sizes are rejected."""
        with pytest.raises(ValueError, match="must be positive"):
            RollingWindowConfig(short=-5)


class TestMissingColumns:
    """Tests for handling DataFrames with missing expected columns."""

    def test_return_features_missing_price(self):
        """Test ReturnFeatures when price column is missing."""
        df = pd.DataFrame({
            "volume": [100, 200, 300],
        }, index=pd.date_range("2024-01-01", periods=3, freq="min", tz="UTC"))

        extractor = ReturnFeatures()
        result = extractor.extract(df)
        # Should handle gracefully
        assert "pct_return" not in result.columns or result["pct_return"].isna().all()

    def test_spread_features_missing_bid_ask(self):
        """Test SpreadFeatures when bid/ask columns are missing."""
        df = pd.DataFrame({
            "price_c": [50.0, 51.0, 52.0],
        }, index=pd.date_range("2024-01-01", periods=3, freq="min", tz="UTC"))

        extractor = SpreadFeatures()
        result = extractor.extract(df)
        # Should handle gracefully
        assert len(result) == 3


class TestDataTypeEdgeCases:
    """Tests for data type edge cases."""

    def test_integer_prices(self):
        """Test handling of integer prices (should work with float operations)."""
        df = pd.DataFrame({
            "price_c": [50, 51, 52],  # integers
            "yes_bid_c": [49, 50, 51],
            "yes_ask_c": [51, 52, 53],
        }, index=pd.date_range("2024-01-01", periods=3, freq="min", tz="UTC"))

        extractor = SpreadFeatures()
        result = extractor.extract(df)

        assert "spread" in result.columns
        assert not result["spread"].isna().all()

    def test_non_datetime_index(self, config):
        """Test validation catches non-datetime index."""
        df = pd.DataFrame({
            "price_c": [50.0, 51.0, 52.0],
        })  # Default integer index

        contract = ContractTimeseries(
            contract_id="BAD_INDEX",
            category="Sports",
            data=df,
        )

        cleaner = DataCleaner(config)
        result = cleaner.validate(df)

        assert not result.is_valid
        assert any("DatetimeIndex" in e for e in result.errors)


class TestMarketDatasetEdgeCases:
    """Tests for MarketDataset edge cases."""

    def test_filter_nonexistent_category(self, sample_dataset):
        """Test filtering by category that doesn't exist."""
        filtered = sample_dataset.filter_by_category("NonExistent")
        assert len(filtered) == 0

    def test_remove_nonexistent_contract(self, sample_dataset):
        """Test removing contract that doesn't exist."""
        # Should not raise
        sample_dataset.remove("NONEXISTENT_ID")
        # Original contracts still present
        assert len(sample_dataset) > 0

    def test_to_panel_empty_dataset(self):
        """Test to_panel on empty dataset."""
        dataset = MarketDataset()
        panel = dataset.to_panel()
        assert len(panel) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
