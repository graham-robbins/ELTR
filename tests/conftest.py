"""
Pytest configuration and shared fixtures for IRP test suite.

This file is automatically loaded by pytest and provides shared fixtures
that can be used across all test modules.
"""

import numpy as np
import pandas as pd
import pytest

from src.utils.config import load_config, IRPConfig
from src.utils.types import ContractTimeseries, MarketDataset


@pytest.fixture
def config() -> IRPConfig:
    """Load default configuration for tests."""
    return load_config()


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n_rows = 100
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


@pytest.fixture
def sample_contract(sample_df) -> ContractTimeseries:
    """Create a sample contract for testing."""
    return ContractTimeseries(
        contract_id="TEST_CONTRACT",
        category="Sports",
        data=sample_df,
    )


@pytest.fixture
def sample_dataset(sample_df) -> MarketDataset:
    """Create a sample dataset with multiple contracts for testing."""
    dataset = MarketDataset()

    for i, category in enumerate(["Sports", "Politics", "Economics"]):
        np.random.seed(42 + i)
        df = sample_df.copy()
        # Add some variation
        df["price_c"] = df["price_c"] + np.random.normal(0, 2, len(df))
        df["volume"] = (df["volume"] * (1 + 0.1 * i)).astype(int)

        contract = ContractTimeseries(
            contract_id=f"TEST_{category.upper()}_{i}",
            category=category,
            data=df,
        )
        dataset.add(contract)

    return dataset


@pytest.fixture
def empty_df() -> pd.DataFrame:
    """Create an empty DataFrame for edge case testing."""
    return pd.DataFrame(
        columns=["price_c", "volume", "yes_bid_c", "yes_ask_c"],
        index=pd.DatetimeIndex([], tz="UTC"),
    )


@pytest.fixture
def single_row_df() -> pd.DataFrame:
    """Create a single-row DataFrame for edge case testing."""
    return pd.DataFrame({
        "price_c": [50.0],
        "volume": [100],
        "yes_bid_c": [49.0],
        "yes_ask_c": [51.0],
    }, index=pd.date_range("2024-01-01", periods=1, freq="min", tz="UTC"))


@pytest.fixture
def df_with_nans() -> pd.DataFrame:
    """Create a DataFrame with NaN values for edge case testing."""
    np.random.seed(42)
    n_rows = 50
    prices = 50 + np.cumsum(np.random.normal(0, 1, n_rows))

    df = pd.DataFrame({
        "price_c": prices,
        "volume": np.random.randint(10, 1000, n_rows).astype(float),
        "yes_bid_c": prices - 1,
        "yes_ask_c": prices + 1,
    }, index=pd.date_range("2024-01-01", periods=n_rows, freq="min", tz="UTC"))

    # Introduce NaNs
    df.loc[df.index[10:15], "price_c"] = np.nan
    df.loc[df.index[20:25], "volume"] = np.nan
    df.loc[df.index[30], "yes_bid_c"] = np.nan

    return df


@pytest.fixture
def df_with_extreme_values() -> pd.DataFrame:
    """Create a DataFrame with extreme values for edge case testing."""
    np.random.seed(42)
    n_rows = 50

    df = pd.DataFrame({
        "price_c": [0.0, 100.0, 50.0] + [50.0] * (n_rows - 3),
        "volume": [0, 1000000, 100] + [100] * (n_rows - 3),
        "yes_bid_c": [0.0, 99.0, 49.0] + [49.0] * (n_rows - 3),
        "yes_ask_c": [1.0, 100.0, 51.0] + [51.0] * (n_rows - 3),
    }, index=pd.date_range("2024-01-01", periods=n_rows, freq="min", tz="UTC"))

    return df
