"""
Unit tests for ingestion module.

Run tests with: pytest tests/test_ingest.py -v
After installing package with: pip install -e .
"""

import pandas as pd
import pytest

from src.utils.config import load_config
from src.utils.types import DataFrequency, MarketDataset
from src.ingest.kalshi_loader import (
    KalshiDataLoader,
    ContractLoader,
    MetadataLoader,
    SchemaValidator,
    TimestampParser,
)


class TestTimestampParser:
    """Tests for TimestampParser class."""

    def test_parse_utc_timestamps(self):
        """Test parsing UTC timestamp strings."""
        parser = TimestampParser()
        series = pd.Series([
            "2024-01-01 12:00:00+00:00",
            "2024-01-02 12:00:00+00:00",
        ])
        result = parser.parse(series)

        assert len(result) == 2
        assert result[0].tzinfo is not None

    def test_parse_handles_invalid(self):
        """Test parsing handles invalid timestamps."""
        parser = TimestampParser()
        series = pd.Series(["invalid", "2024-01-01 12:00:00+00:00", None])
        result = parser.parse(series)

        assert pd.isna(result[0])
        assert not pd.isna(result[1])
        assert pd.isna(result[2])

    def test_timezone_conversion(self):
        """Test timezone conversion to Eastern."""
        parser = TimestampParser(output_tz="America/New_York")
        series = pd.Series(["2024-01-01 17:00:00+00:00"])  # 5 PM UTC
        result = parser.parse(series)
        eastern = parser.to_eastern(result)

        assert eastern[0].hour == 12  # 12 PM Eastern in winter


class TestSchemaValidator:
    """Tests for SchemaValidator class."""

    def test_validate_timeseries_columns(self):
        """Test timeseries DataFrame validation."""
        config = load_config()
        validator = SchemaValidator(config.data.schema_path)

        df = pd.DataFrame({
            "datetime": ["2024-01-01"],
            "price_c": [50.0],
            "yes_bid_c": [49.0],
            "yes_ask_c": [51.0],
            "volume": [100],
            "contract_id": ["TEST"],
        })

        is_valid, missing = validator.validate_timeseries(df)
        assert isinstance(is_valid, bool)
        assert isinstance(missing, list)


class TestContractLoader:
    """Tests for ContractLoader class."""

    def test_detect_frequency_minute(self):
        """Test frequency detection for minute data."""
        config = load_config()
        parser = TimestampParser()
        loader = ContractLoader(config, parser)

        df = pd.DataFrame({
            "price_c": [50.0, 51.0, 52.0],
        }, index=pd.date_range("2024-01-01", periods=3, freq="min", tz="UTC"))

        freq = loader._detect_frequency(df)
        assert freq == DataFrequency.MINUTE

    def test_detect_frequency_daily(self):
        """Test frequency detection for daily data."""
        config = load_config()
        parser = TimestampParser()
        loader = ContractLoader(config, parser)

        df = pd.DataFrame({
            "price_c": [50.0, 51.0, 52.0],
        }, index=pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"))

        freq = loader._detect_frequency(df)
        assert freq == DataFrequency.DAY


class TestKalshiDataLoader:
    """Tests for main KalshiDataLoader class."""

    def test_loader_initialization(self):
        """Test loader initializes correctly."""
        config = load_config()
        loader = KalshiDataLoader(config)

        assert loader.config is not None
        assert loader.validator is not None
        assert loader.timestamp_parser is not None

    def test_get_available_contracts(self):
        """Test listing available contracts."""
        config = load_config()
        loader = KalshiDataLoader(config)

        available = loader.get_available_contracts()

        assert isinstance(available, pd.DataFrame)
        if not available.empty:
            assert "contract_id" in available.columns
            assert "category" in available.columns


class TestMarketDataset:
    """Tests for MarketDataset container."""

    def test_empty_dataset(self):
        """Test empty dataset properties."""
        dataset = MarketDataset()

        assert len(dataset) == 0
        assert dataset.categories == []
        assert dataset.contract_ids == []

    def test_filter_by_category(self):
        """Test category filtering."""
        from src.utils.types import ContractTimeseries

        dataset = MarketDataset()

        ts1 = ContractTimeseries(
            contract_id="TEST1",
            category="Sports",
            data=pd.DataFrame({"price_c": [50.0]}),
        )
        ts2 = ContractTimeseries(
            contract_id="TEST2",
            category="Politics",
            data=pd.DataFrame({"price_c": [60.0]}),
        )

        dataset.add(ts1)
        dataset.add(ts2)

        sports = dataset.filter_by_category("Sports")
        assert len(sports) == 1
        assert "TEST1" in sports.contract_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
