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
    LifecycleFeatures,
    MicrostructureRegimeFeatures,
    MicrostructureState,
)
from src.utils.time_binning import compute_lifecycle_features
from src.microstructure.regimes import compute_microstructure_regime
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


class TestPostResolutionFiltering:
    """Tests for post-resolution observation filtering."""

    def test_exclude_post_resolution_default(self):
        """Test that post-resolution observations are excluded by default."""
        df = pd.DataFrame({
            "price_c": [50.0] * 10,
            "volume": [100] * 10,
        }, index=pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC"))

        # Settlement at hour 5, so hours 6-9 are post-resolution
        settlement_time = df.index[5]

        result = compute_lifecycle_features(
            df,
            settlement_time=settlement_time,
            exclude_post_resolution=True,  # Default
        )

        # Should only have observations up to and including settlement
        assert len(result) <= 6
        assert (result["tts_hours"] >= 0).all()

    def test_exclude_post_resolution_disabled(self):
        """Test that post-resolution observations are kept when disabled."""
        df = pd.DataFrame({
            "price_c": [50.0] * 10,
            "volume": [100] * 10,
        }, index=pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC"))

        # Settlement at hour 5
        settlement_time = df.index[5]

        result = compute_lifecycle_features(
            df,
            settlement_time=settlement_time,
            exclude_post_resolution=False,  # Keep all data
        )

        # Should keep all observations
        assert len(result) == 10
        # Should have some negative tts_hours (post-resolution)
        assert (result["tts_hours"] < 0).any()

    def test_no_filtering_when_all_pre_resolution(self):
        """Test no filtering when all observations are pre-resolution."""
        df = pd.DataFrame({
            "price_c": [50.0] * 10,
            "volume": [100] * 10,
        }, index=pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC"))

        # Settlement at last observation (default behavior)
        result = compute_lifecycle_features(df)

        # All observations should be kept
        assert len(result) == 10
        assert (result["tts_hours"] >= 0).all()

    def test_lifecycle_features_empty_after_filtering(self):
        """Test handling when all observations would be filtered."""
        df = pd.DataFrame({
            "price_c": [50.0] * 5,
            "volume": [100] * 5,
        }, index=pd.date_range("2024-01-01 10:00", periods=5, freq="h", tz="UTC"))

        # Settlement before all observations
        settlement_time = pd.Timestamp("2024-01-01 05:00", tz="UTC")

        result = compute_lifecycle_features(
            df,
            settlement_time=settlement_time,
            exclude_post_resolution=True,
        )

        # Should result in empty DataFrame
        assert len(result) == 0


class TestZeroReturnFrozenState:
    """Tests for zero-return FROZEN state condition."""

    def test_frozen_on_zero_return(self):
        """Test that zero pct_return triggers FROZEN state."""
        df = pd.DataFrame({
            "price_c": [50.0, 50.0, 50.0, 50.0, 50.0],
            "volume": [100, 100, 100, 100, 100],
            "pct_return": [0.01, 0.0, 0.0, 0.0, 0.01],
            "spread_pct": [0.02, 0.02, 0.02, 0.02, 0.02],
        }, index=pd.date_range("2024-01-01", periods=5, freq="min", tz="UTC"))

        result = compute_microstructure_regime(df)

        # Rows with zero return should be FROZEN
        assert result.iloc[1]["microstructure_state"] == MicrostructureState.FROZEN.value
        assert result.iloc[2]["microstructure_state"] == MicrostructureState.FROZEN.value
        assert result.iloc[3]["microstructure_state"] == MicrostructureState.FROZEN.value

    def test_frozen_zero_return_overrides_other_states(self):
        """Test that zero return FROZEN has highest priority."""
        df = pd.DataFrame({
            "price_c": [50.0] * 10,
            "volume": [1000] * 10,  # High volume (not low volume frozen)
            "pct_return": [0.0] * 10,  # All zero returns
            "spread_pct": [0.20] * 10,  # Wide spread (would be THIN)
        }, index=pd.date_range("2024-01-01", periods=10, freq="min", tz="UTC"))

        result = compute_microstructure_regime(df)

        # All should be FROZEN because zero return has highest priority
        assert (result["microstructure_state"] == MicrostructureState.FROZEN.value).all()

    def test_non_zero_return_not_frozen(self):
        """Test that non-zero return does not trigger FROZEN state."""
        df = pd.DataFrame({
            "price_c": [50.0, 51.0, 52.0, 53.0, 54.0],
            "volume": [100, 100, 100, 100, 100],
            "pct_return": [0.02, 0.02, 0.02, 0.02, 0.02],
            "spread_pct": [0.02, 0.02, 0.02, 0.02, 0.02],
        }, index=pd.date_range("2024-01-01", periods=5, freq="min", tz="UTC"))

        result = compute_microstructure_regime(df)

        # No rows should be FROZEN (normal return and volume)
        # They should be NORMAL since no other conditions are met
        assert not (result["microstructure_state"] == MicrostructureState.FROZEN.value).any()


class TestMicrostructureRegimeEdgeCases:
    """Edge case tests for microstructure regime classification."""

    def test_regime_empty_dataframe(self, empty_df):
        """Test regime classification on empty DataFrame."""
        result = compute_microstructure_regime(empty_df)
        assert len(result) == 0

    def test_regime_single_row(self, single_row_df):
        """Test regime classification on single-row DataFrame."""
        single_row_df["pct_return"] = np.nan
        single_row_df["spread_pct"] = 0.02

        result = compute_microstructure_regime(single_row_df)

        assert len(result) == 1
        assert "microstructure_state" in result.columns

    def test_regime_all_nan_returns(self):
        """Test regime classification when all returns are NaN."""
        df = pd.DataFrame({
            "price_c": [50.0] * 5,
            "volume": [100] * 5,
            "pct_return": [np.nan] * 5,
            "spread_pct": [0.02] * 5,
        }, index=pd.date_range("2024-01-01", periods=5, freq="min", tz="UTC"))

        result = compute_microstructure_regime(df)

        # Should not crash, states should be assigned
        assert len(result) == 5
        assert "microstructure_state" in result.columns

    def test_regime_mixed_zero_and_nan_returns(self):
        """Test regime with mix of zero and NaN returns."""
        df = pd.DataFrame({
            "price_c": [50.0] * 6,
            "volume": [100] * 6,
            "pct_return": [0.01, 0.0, np.nan, 0.0, 0.01, np.nan],
            "spread_pct": [0.02] * 6,
        }, index=pd.date_range("2024-01-01", periods=6, freq="min", tz="UTC"))

        result = compute_microstructure_regime(df)

        # Zero return rows should be FROZEN
        assert result.iloc[1]["microstructure_state"] == MicrostructureState.FROZEN.value
        assert result.iloc[3]["microstructure_state"] == MicrostructureState.FROZEN.value
        # Non-zero, non-NaN rows should not be FROZEN
        assert result.iloc[0]["microstructure_state"] != MicrostructureState.FROZEN.value
        assert result.iloc[4]["microstructure_state"] != MicrostructureState.FROZEN.value


class TestLifecycleEdgeCases:
    """Edge case tests for lifecycle feature computation."""

    def test_lifecycle_empty_dataframe(self, empty_df):
        """Test lifecycle computation on empty DataFrame."""
        result = compute_lifecycle_features(empty_df)
        assert len(result) == 0

    def test_lifecycle_single_row(self, single_row_df):
        """Test lifecycle computation on single-row DataFrame."""
        result = compute_lifecycle_features(single_row_df)

        # Single row: lifecycle_ratio should be 0 or 1
        assert len(result) == 1
        assert "lifecycle_ratio" in result.columns

    def test_lifecycle_features_extractor_empty_df(self, empty_df):
        """Test LifecycleFeatures extractor on empty DataFrame."""
        extractor = LifecycleFeatures()
        result = extractor.extract(empty_df)
        assert len(result) == 0

    def test_lifecycle_features_extractor_single_row(self, single_row_df):
        """Test LifecycleFeatures extractor on single-row DataFrame."""
        extractor = LifecycleFeatures()
        result = extractor.extract(single_row_df)
        assert len(result) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
