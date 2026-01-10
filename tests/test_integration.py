"""
Integration tests for the IRP pipeline.

These tests verify the full pipeline flow from data processing
through feature engineering and microstructure analysis.

Run tests with: pytest tests/test_integration.py -v
After installing package with: pip install -e .
"""

import numpy as np
import pandas as pd
import pytest

from src.utils.config import load_config
from src.utils.types import ContractTimeseries, MarketDataset
from src.clean.cleaner import DataCleaner
from src.features.feature_engineering import FeatureEngineer
from src.microstructure.analysis import MicrostructureAnalyzer


class TestFullPipeline:
    """Integration tests for the complete pipeline."""

    def test_pipeline_single_contract(self, sample_contract, config):
        """Test full pipeline on a single contract."""
        # Clean
        cleaner = DataCleaner(config)
        cleaned_contract, clean_stats = cleaner.clean_contract(sample_contract)

        assert cleaned_contract.n_observations > 0
        assert clean_stats.original_rows == sample_contract.n_observations

        # Feature engineering
        engineer = FeatureEngineer(config)
        featured_contract = engineer.engineer_features(cleaned_contract)

        assert "pct_return" in featured_contract.data.columns
        assert "volatility_short" in featured_contract.data.columns
        assert "spread" in featured_contract.data.columns

        # Microstructure analysis
        analyzer = MicrostructureAnalyzer(config, n_jobs=1)
        metrics = analyzer.analyze_contract(featured_contract)

        assert metrics.contract_id == sample_contract.contract_id
        assert not np.isnan(metrics.avg_spread)
        assert not np.isnan(metrics.avg_volume)

    def test_pipeline_dataset(self, sample_dataset, config):
        """Test full pipeline on a dataset with multiple contracts."""
        # Clean
        cleaner = DataCleaner(config)
        cleaned_dataset, clean_stats = cleaner.clean_dataset(sample_dataset)

        assert len(cleaned_dataset) > 0
        assert len(clean_stats) > 0

        # Feature engineering
        engineer = FeatureEngineer(config)
        featured_dataset = engineer.engineer_dataset(cleaned_dataset)

        assert len(featured_dataset) == len(cleaned_dataset)
        for contract in featured_dataset:
            assert "pct_return" in contract.data.columns

        # Microstructure analysis
        analyzer = MicrostructureAnalyzer(config, n_jobs=1)
        all_metrics, summary_df = analyzer.analyze_dataset(featured_dataset)

        assert len(all_metrics) == len(featured_dataset)
        assert len(summary_df) == len(featured_dataset)
        assert "contract_id" in summary_df.columns

    def test_pipeline_with_empty_contract_filtered(self, config):
        """Test that contracts with too few observations are filtered."""
        config.cleaning.min_observations = 10

        # Create contract with very few observations
        df = pd.DataFrame({
            "price_c": [50.0, 51.0, 52.0],
            "volume": [100, 150, 200],
            "yes_bid_c": [49.0, 50.0, 51.0],
            "yes_ask_c": [51.0, 52.0, 53.0],
        }, index=pd.date_range("2024-01-01", periods=3, freq="min", tz="UTC"))

        contract = ContractTimeseries(
            contract_id="SMALL_CONTRACT",
            category="Sports",
            data=df,
        )

        dataset = MarketDataset()
        dataset.add(contract)

        cleaner = DataCleaner(config)
        cleaned_dataset, _ = cleaner.clean_dataset(dataset)

        # Contract should be filtered out due to insufficient observations
        assert len(cleaned_dataset) == 0

    def test_pipeline_preserves_metadata(self, sample_contract, config):
        """Test that contract metadata is preserved through the pipeline."""
        # Clean
        cleaner = DataCleaner(config)
        cleaned_contract, _ = cleaner.clean_contract(sample_contract)

        assert cleaned_contract.contract_id == sample_contract.contract_id
        assert cleaned_contract.category == sample_contract.category

        # Feature engineering
        engineer = FeatureEngineer(config)
        featured_contract = engineer.engineer_features(cleaned_contract)

        assert featured_contract.contract_id == sample_contract.contract_id
        assert featured_contract.category == sample_contract.category


class TestCleaningIntegration:
    """Integration tests for the cleaning module."""

    def test_cleaning_handles_missing_values(self, df_with_nans, config):
        """Test that cleaning properly handles missing values."""
        contract = ContractTimeseries(
            contract_id="NAN_CONTRACT",
            category="Sports",
            data=df_with_nans,
        )

        cleaner = DataCleaner(config)
        cleaned_contract, stats = cleaner.clean_contract(contract)

        # Should have imputed some values
        assert stats.nulls_imputed > 0
        # Price column should have fewer NaNs after cleaning
        assert cleaned_contract.data["price_c"].isna().sum() < df_with_nans["price_c"].isna().sum()

    def test_cleaning_handles_extreme_values(self, df_with_extreme_values, config):
        """Test that cleaning properly handles extreme values."""
        contract = ContractTimeseries(
            contract_id="EXTREME_CONTRACT",
            category="Sports",
            data=df_with_extreme_values,
        )

        cleaner = DataCleaner(config)
        cleaned_contract, stats = cleaner.clean_contract(contract)

        # Prices should be clipped to valid range
        assert cleaned_contract.data["price_c"].min() >= config.cleaning.min_price
        assert cleaned_contract.data["price_c"].max() <= config.cleaning.max_price


class TestFeatureEngineeringIntegration:
    """Integration tests for feature engineering."""

    def test_all_features_computed(self, sample_contract, config):
        """Test that all expected features are computed."""
        engineer = FeatureEngineer(config)
        featured_contract = engineer.engineer_features(sample_contract)

        expected_features = [
            "pct_return",
            "log_return",
            "volatility_short",
            "volume_ma",
            "spread",
            "midpoint",
        ]

        for feature in expected_features:
            assert feature in featured_contract.data.columns, f"Missing feature: {feature}"

    def test_feature_values_valid(self, sample_contract, config):
        """Test that computed feature values are valid."""
        engineer = FeatureEngineer(config)
        featured_contract = engineer.engineer_features(sample_contract)

        # Returns should have first value NaN (no previous price)
        assert pd.isna(featured_contract.data["pct_return"].iloc[0])

        # Spreads should be non-negative
        spread_valid = featured_contract.data["spread"].dropna()
        assert (spread_valid >= 0).all()

        # Midpoint should be between bid and ask
        df = featured_contract.data
        valid_mask = df["midpoint"].notna() & df["yes_bid_c"].notna() & df["yes_ask_c"].notna()
        if valid_mask.any():
            assert (df.loc[valid_mask, "midpoint"] >= df.loc[valid_mask, "yes_bid_c"]).all()
            assert (df.loc[valid_mask, "midpoint"] <= df.loc[valid_mask, "yes_ask_c"]).all()


class TestMicrostructureIntegration:
    """Integration tests for microstructure analysis."""

    def test_analysis_produces_metrics(self, sample_contract, config):
        """Test that microstructure analysis produces valid metrics."""
        # First add features
        engineer = FeatureEngineer(config)
        featured_contract = engineer.engineer_features(sample_contract)

        # Then analyze
        analyzer = MicrostructureAnalyzer(config, n_jobs=1)
        metrics = analyzer.analyze_contract(featured_contract)

        # Check key metrics are present and valid
        assert metrics.contract_id == sample_contract.contract_id
        assert isinstance(metrics.avg_spread, float)
        assert isinstance(metrics.avg_volume, float)
        assert isinstance(metrics.surge_count, int)
        assert metrics.surge_count >= 0

    def test_category_aggregates(self, sample_dataset, config):
        """Test category aggregate computation."""
        # Clean and feature engineer
        cleaner = DataCleaner(config)
        cleaned_dataset, _ = cleaner.clean_dataset(sample_dataset)

        engineer = FeatureEngineer(config)
        featured_dataset = engineer.engineer_dataset(cleaned_dataset)

        # Compute aggregates
        analyzer = MicrostructureAnalyzer(config, n_jobs=1)
        category_df = analyzer.compute_category_aggregates(featured_dataset)

        assert len(category_df) > 0
        assert "category" in category_df.columns
        assert "n_contracts" in category_df.columns
        assert "avg_spread" in category_df.columns


class TestParallelProcessing:
    """Tests for parallel processing functionality."""

    def test_parallel_vs_sequential_same_results(self, sample_dataset, config):
        """Test that parallel and sequential processing produce same results."""
        # Clean and feature engineer
        cleaner = DataCleaner(config)
        cleaned_dataset, _ = cleaner.clean_dataset(sample_dataset)

        engineer = FeatureEngineer(config)
        featured_dataset = engineer.engineer_dataset(cleaned_dataset)

        # Sequential
        analyzer_seq = MicrostructureAnalyzer(config, n_jobs=1)
        metrics_seq, df_seq = analyzer_seq.analyze_dataset(featured_dataset)

        # Parallel (if available)
        try:
            from joblib import Parallel
            analyzer_par = MicrostructureAnalyzer(config, n_jobs=2)
            metrics_par, df_par = analyzer_par.analyze_dataset(featured_dataset)

            # Results should be equivalent
            assert len(metrics_seq) == len(metrics_par)
            for m_seq, m_par in zip(metrics_seq, metrics_par):
                assert m_seq.contract_id == m_par.contract_id
                np.testing.assert_almost_equal(m_seq.avg_spread, m_par.avg_spread, decimal=10)

        except ImportError:
            pytest.skip("joblib not available for parallel testing")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
