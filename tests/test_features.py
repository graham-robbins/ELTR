"""
Unit tests for feature engineering module.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.types import ContractTimeseries, MarketDataset
from src.features.feature_engineering import (
    FeatureEngineer,
    ReturnFeatures,
    VolatilityFeatures,
    LiquidityFeatures,
    SpreadFeatures,
    DepthFeatures,
    RegimeFeatures,
)


def create_sample_df(n_rows: int = 100) -> pd.DataFrame:
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    prices = 50 + np.cumsum(np.random.normal(0, 1, n_rows))
    prices = np.clip(prices, 1, 99)

    return pd.DataFrame({
        "price_c": prices,
        "price_o": prices - np.random.uniform(0, 1, n_rows),
        "price_h": prices + np.random.uniform(0, 2, n_rows),
        "price_l": prices - np.random.uniform(0, 2, n_rows),
        "volume": np.random.randint(10, 1000, n_rows),
        "trade_count": np.random.randint(1, 50, n_rows),
        "yes_bid_c": prices - np.random.uniform(0.5, 2, n_rows),
        "yes_ask_c": prices + np.random.uniform(0.5, 2, n_rows),
        "yes_bid_h": prices - np.random.uniform(0, 1, n_rows),
        "yes_bid_l": prices - np.random.uniform(1, 3, n_rows),
        "yes_ask_h": prices + np.random.uniform(1, 3, n_rows),
        "yes_ask_l": prices + np.random.uniform(0, 1, n_rows),
    }, index=pd.date_range("2024-01-01", periods=n_rows, freq="min", tz="UTC"))


class TestReturnFeatures:
    """Tests for ReturnFeatures extractor."""

    def test_pct_return(self):
        """Test percent return calculation."""
        extractor = ReturnFeatures()
        df = create_sample_df(10)

        result = extractor.extract(df)

        assert "pct_return" in result.columns
        assert pd.isna(result["pct_return"].iloc[0])
        assert not result["pct_return"].iloc[1:].isna().all()

    def test_log_return(self):
        """Test log return calculation."""
        extractor = ReturnFeatures()
        df = create_sample_df(10)

        result = extractor.extract(df)

        assert "log_return" in result.columns

    def test_logit_return(self):
        """Test logit return calculation."""
        extractor = ReturnFeatures()
        df = create_sample_df(10)

        result = extractor.extract(df)

        assert "logit_return" in result.columns
        assert result["logit_return"].notna().sum() > 0

    def test_abs_return(self):
        """Test absolute return calculation."""
        extractor = ReturnFeatures()
        df = create_sample_df(10)

        result = extractor.extract(df)

        assert "abs_return" in result.columns
        valid_abs = result["abs_return"].dropna()
        assert (valid_abs >= 0).all()


class TestVolatilityFeatures:
    """Tests for VolatilityFeatures extractor."""

    def test_rolling_volatility(self):
        """Test rolling volatility calculation."""
        extractor = VolatilityFeatures(
            windows={"short": 5, "long": 20},
            min_periods=3,
        )
        df = create_sample_df(50)
        df = ReturnFeatures().extract(df)

        result = extractor.extract(df)

        assert "volatility_short" in result.columns
        assert "volatility_long" in result.columns

    def test_annualized_volatility(self):
        """Test annualized volatility."""
        extractor = VolatilityFeatures(
            windows={"short": 5},
            annualization_factor=252,
        )
        df = create_sample_df(20)
        df = ReturnFeatures().extract(df)

        result = extractor.extract(df)

        assert "volatility_short_ann" in result.columns

    def test_parkinson_volatility(self):
        """Test Parkinson volatility estimator."""
        extractor = VolatilityFeatures()
        df = create_sample_df(30)
        df = ReturnFeatures().extract(df)

        result = extractor.extract(df)

        assert "parkinson_vol" in result.columns

    def test_price_range(self):
        """Test price range calculation."""
        extractor = VolatilityFeatures()
        df = create_sample_df(20)
        df = ReturnFeatures().extract(df)

        result = extractor.extract(df)

        assert "price_range" in result.columns
        valid_range = result["price_range"].dropna()
        assert (valid_range >= 0).all()


class TestLiquidityFeatures:
    """Tests for LiquidityFeatures extractor."""

    def test_volume_ma(self):
        """Test volume moving average."""
        extractor = LiquidityFeatures(volume_ma_window=10)
        df = create_sample_df(20)

        result = extractor.extract(df)

        assert "volume_ma" in result.columns
        assert result["volume_ma"].notna().sum() > 0

    def test_volume_zscore(self):
        """Test volume z-score."""
        extractor = LiquidityFeatures()
        df = create_sample_df(50)

        result = extractor.extract(df)

        assert "volume_zscore" in result.columns

    def test_volume_surge(self):
        """Test volume surge ratio."""
        extractor = LiquidityFeatures()
        df = create_sample_df(50)

        result = extractor.extract(df)

        assert "volume_surge" in result.columns

    def test_avg_trade_size(self):
        """Test average trade size."""
        extractor = LiquidityFeatures()
        df = create_sample_df(20)

        result = extractor.extract(df)

        assert "avg_trade_size" in result.columns


class TestSpreadFeatures:
    """Tests for SpreadFeatures extractor."""

    def test_spread_calculation(self):
        """Test spread calculation."""
        extractor = SpreadFeatures()
        df = create_sample_df(20)

        result = extractor.extract(df)

        assert "spread" in result.columns
        valid_spread = result["spread"].dropna()
        assert (valid_spread >= 0).all()

    def test_midpoint(self):
        """Test midpoint calculation."""
        extractor = SpreadFeatures()
        df = create_sample_df(20)

        result = extractor.extract(df)

        assert "midpoint" in result.columns

    def test_spread_pct(self):
        """Test spread percentage."""
        extractor = SpreadFeatures()
        df = create_sample_df(20)

        result = extractor.extract(df)

        assert "spread_pct" in result.columns

    def test_spread_bps(self):
        """Test spread in basis points."""
        extractor = SpreadFeatures(basis_points=True)
        df = create_sample_df(20)

        result = extractor.extract(df)

        assert "spread_bps" in result.columns

    def test_effective_spread(self):
        """Test effective spread calculation."""
        extractor = SpreadFeatures()
        df = create_sample_df(20)

        result = extractor.extract(df)

        assert "effective_spread" in result.columns


class TestDepthFeatures:
    """Tests for DepthFeatures extractor."""

    def test_bid_range(self):
        """Test bid range calculation."""
        extractor = DepthFeatures()
        df = create_sample_df(20)

        result = extractor.extract(df)

        assert "bid_range" in result.columns

    def test_ask_range(self):
        """Test ask range calculation."""
        extractor = DepthFeatures()
        df = create_sample_df(20)

        result = extractor.extract(df)

        assert "ask_range" in result.columns

    def test_book_thinning(self):
        """Test book thinning metric."""
        extractor = DepthFeatures()
        df = create_sample_df(20)

        result = extractor.extract(df)

        assert "book_thinning" in result.columns


class TestRegimeFeatures:
    """Tests for RegimeFeatures extractor."""

    def test_hour_of_day(self):
        """Test hour of day extraction."""
        extractor = RegimeFeatures()
        df = create_sample_df(20)

        result = extractor.extract(df)

        assert "hour_of_day" in result.columns
        assert result["hour_of_day"].min() >= 0
        assert result["hour_of_day"].max() <= 23

    def test_day_of_week(self):
        """Test day of week extraction."""
        extractor = RegimeFeatures()
        df = create_sample_df(20)

        result = extractor.extract(df)

        assert "day_of_week" in result.columns
        assert result["day_of_week"].min() >= 0
        assert result["day_of_week"].max() <= 6

    def test_is_market_hours(self):
        """Test market hours flag."""
        extractor = RegimeFeatures()
        df = create_sample_df(20)

        result = extractor.extract(df)

        assert "is_market_hours" in result.columns
        assert result["is_market_hours"].isin([0, 1]).all()


class TestFeatureEngineer:
    """Tests for main FeatureEngineer class."""

    def test_engineer_contract(self):
        """Test feature engineering for single contract."""
        config = load_config()
        engineer = FeatureEngineer(config)

        df = create_sample_df(100)
        contract = ContractTimeseries(
            contract_id="TEST",
            category="Sports",
            data=df,
        )

        result = engineer.engineer_features(contract)

        assert result.n_observations == 100
        assert "pct_return" in result.data.columns
        assert "volatility_short" in result.data.columns
        assert "spread" in result.data.columns

    def test_engineer_dataset(self):
        """Test feature engineering for dataset."""
        config = load_config()
        engineer = FeatureEngineer(config)

        dataset = MarketDataset()
        for i in range(3):
            df = create_sample_df(50)
            contract = ContractTimeseries(
                contract_id=f"TEST_{i}",
                category="Sports",
                data=df,
            )
            dataset.add(contract)

        result = engineer.engineer_dataset(dataset)

        assert len(result) == 3
        for contract in result:
            assert "pct_return" in contract.data.columns

    def test_get_feature_names(self):
        """Test getting list of feature names."""
        config = load_config()
        engineer = FeatureEngineer(config)

        feature_names = engineer.get_feature_names()

        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        assert "pct_return" in feature_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
