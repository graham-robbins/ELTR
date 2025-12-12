"""
Kalshi market data ingestion module.

Handles loading, parsing, and standardization of Kalshi prediction market
data from CSV files into canonical DataFrame format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from src.utils.config import IRPConfig, get_config
from src.utils.logging import get_logger
from src.utils.types import (
    Category,
    ContractID,
    ContractMetadata,
    ContractTimeseries,
    DataFrequency,
    MarketDataset,
)

logger = get_logger("ingest")


# Category canonicalization mapping (Section 8)
CATEGORY_ALIASES = {
    # Sports
    "sports": "Sports",
    "sport": "Sports",
    "SPORTS": "Sports",
    "nfl": "Sports",
    "nba": "Sports",
    "mlb": "Sports",
    # Politics
    "politics": "Politics",
    "political": "Politics",
    "POLITICS": "Politics",
    "election": "Politics",
    "elections": "Politics",
    # Economics
    "economics": "Economics",
    "economic": "Economics",
    "econ": "Economics",
    "ECONOMICS": "Economics",
    "fed": "Economics",
    "inflation": "Economics",
    "cpi": "Economics",
    # Crypto
    "crypto": "Crypto",
    "cryptocurrency": "Crypto",
    "CRYPTO": "Crypto",
    "bitcoin": "Crypto",
    "btc": "Crypto",
    # Weather
    "weather": "Weather",
    "WEATHER": "Weather",
    "climate": "Weather",
}

# Canonical category list
CANONICAL_CATEGORIES = ["Sports", "Politics", "Economics", "Crypto", "Weather"]


def canonical_category(category: str) -> str:
    """
    Canonicalize category name (Section 8).

    Parameters
    ----------
    category : str
        Input category name (may be alias or variant).

    Returns
    -------
    str
        Canonical category name.
    """
    if category in CANONICAL_CATEGORIES:
        return category

    normalized = category.lower().strip()
    if normalized in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[normalized]

    # Title-case fallback
    return category.title()


def infer_event_time(
    contract: "ContractTimeseries",
    metadata: "ContractMetadata | None" = None,
) -> pd.Timestamp | None:
    """
    Infer event time based on category-specific rules (Section 2).

    Parameters
    ----------
    contract : ContractTimeseries
        Contract data.
    metadata : ContractMetadata | None
        Contract metadata (may have end_date).

    Returns
    -------
    pd.Timestamp | None
        Inferred event time, or None if cannot infer.
    """
    category = canonical_category(contract.category)

    # Rule 1: Use metadata end_date if available
    if metadata is not None and metadata.end_date is not None:
        return pd.Timestamp(metadata.end_date)

    # Rule 2: Category-specific inference
    df = contract.data
    if df.empty:
        return None

    data_end = df.index.max()

    if category == "Sports":
        # Sports: event_time = scheduled game start
        # If no metadata, use first observation (games start before contract activity)
        return df.index.min()

    elif category == "Economics":
        # Economics: event_time = data release time
        # Typically 8:30 AM ET for major releases
        # Infer as last observation time (market closes at release)
        return data_end

    elif category == "Politics":
        # Politics: event_time = announcement/vote time
        # Use last observation
        return data_end

    elif category == "Crypto":
        # Crypto: typically daily close prices, use end of data
        return data_end

    elif category == "Weather":
        # Weather: forecast verification time
        # Use last observation
        return data_end

    # Default: use last observation
    return data_end


# Canonical column schema for timeseries data
CANONICAL_COLUMNS = [
    "timestamp",
    "contract_id",
    "category",
    "price_o",
    "price_h",
    "price_l",
    "price_c",
    "volume",
    "trade_count",
    "yes_bid_o",
    "yes_bid_h",
    "yes_bid_l",
    "yes_bid_c",
    "yes_ask_o",
    "yes_ask_h",
    "yes_ask_l",
    "yes_ask_c",
]

# Expected raw columns from Kalshi CSV export
RAW_COLUMNS = [
    "datetime",
    "price_c",
    "yes_bid_c",
    "yes_ask_c",
    "volume",
    "price_o",
    "price_h",
    "price_l",
    "yes_bid_o",
    "yes_bid_h",
    "yes_bid_l",
    "yes_ask_o",
    "yes_ask_h",
    "yes_ask_l",
    "trade_count",
    "contract_id",
]


class SchemaValidator:
    """
    Validates data against schema definitions.

    Ensures loaded data conforms to expected structure defined
    in schema_mapping.json.
    """

    def __init__(self, schema_path: Path):
        """
        Initialize validator with schema file.

        Parameters
        ----------
        schema_path : Path
            Path to schema_mapping.json file.
        """
        self.schema_path = schema_path
        self.schema = self._load_schema()

    def _load_schema(self) -> dict:
        """Load schema from JSON file."""
        if not self.schema_path.exists():
            logger.warning(f"Schema file not found: {self.schema_path}")
            return {}

        with open(self.schema_path) as f:
            return json.load(f)

    @property
    def timeseries_fields(self) -> list[str]:
        """Expected timeseries CSV fields."""
        return list(self.schema.get("timeseries_csv", {}).keys())

    @property
    def metadata_fields(self) -> list[str]:
        """Expected metadata CSV fields."""
        return list(self.schema.get("metadata_csv", {}).keys())

    def validate_timeseries(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        """
        Validate timeseries DataFrame against schema.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate.

        Returns
        -------
        tuple[bool, list[str]]
            (is_valid, list of missing columns)
        """
        expected = set(self.timeseries_fields)
        actual = set(df.columns)
        missing = expected - actual
        return len(missing) == 0, list(missing)

    def validate_metadata(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        """
        Validate metadata DataFrame against schema.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate.

        Returns
        -------
        tuple[bool, list[str]]
            (is_valid, list of missing columns)
        """
        expected = set(self.metadata_fields)
        actual = set(df.columns)
        missing = expected - actual
        return len(missing) == 0, list(missing)


class TimestampParser:
    """
    Handles timestamp parsing and timezone conversion.

    Forces UTC normalization during ingestion (Section 16).
    Converts raw datetime strings to standardized pandas Timestamps
    with proper timezone handling.
    """

    def __init__(self, input_tz: str = "UTC", output_tz: str = "UTC"):
        """
        Initialize parser with timezone settings.

        Parameters
        ----------
        input_tz : str
            Input timezone (default UTC).
        output_tz : str
            Output timezone for storage. FORCED TO UTC (Section 16).
        """
        self.input_tz = input_tz
        # Force UTC for all internal storage (Section 16)
        self.output_tz = "UTC"

    def parse(self, series: pd.Series) -> pd.DatetimeIndex:
        """
        Parse datetime strings to DatetimeIndex in UTC.

        Forces UTC normalization per Section 16.

        Parameters
        ----------
        series : pd.Series
            Series of datetime strings.

        Returns
        -------
        pd.DatetimeIndex
            Parsed timestamps in UTC.
        """
        # Step 1: Coerce to datetime64 with errors='coerce' (invalid -> NaT)
        parsed = pd.to_datetime(series, errors="coerce")

        # Step 2: Convert to DatetimeIndex for proper tz handling
        dt_index = pd.DatetimeIndex(parsed)

        # Step 3: Normalize to UTC
        return self._normalize_to_utc(dt_index)

    def _normalize_to_utc(self, dt_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """
        Normalize DatetimeIndex to UTC (Section 16).

        Handles both naive and tz-aware inputs.

        Parameters
        ----------
        dt_index : pd.DatetimeIndex
            Input DatetimeIndex (may be naive or tz-aware).

        Returns
        -------
        pd.DatetimeIndex
            UTC-normalized DatetimeIndex.
        """
        if dt_index.tz is None:
            # Naive timestamps: assume UTC and localize
            return dt_index.tz_localize("UTC")
        else:
            # Already tz-aware: convert to UTC
            return dt_index.tz_convert("UTC")

    def to_eastern(self, timestamps: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Convert timestamps to Eastern Time for display."""
        return timestamps.tz_convert("America/New_York")

    def to_utc(self, timestamps: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """
        Ensure timestamps are in UTC (Section 16).

        This is the SINGLE NORMALIZATION POINT for all timestamps.
        """
        return self._normalize_to_utc(timestamps)


class ContractLoader:
    """
    Loads individual contract CSV files.

    Handles parsing, validation, and standardization of single
    contract timeseries data.
    """

    def __init__(
        self,
        config: IRPConfig,
        timestamp_parser: TimestampParser,
        validator: SchemaValidator | None = None,
    ):
        """
        Initialize contract loader.

        Parameters
        ----------
        config : IRPConfig
            Platform configuration.
        timestamp_parser : TimestampParser
            Timestamp parsing handler.
        validator : SchemaValidator | None
            Optional schema validator.
        """
        self.config = config
        self.timestamp_parser = timestamp_parser
        self.validator = validator

    def load(self, filepath: Path, category: Category) -> ContractTimeseries | None:
        """
        Load single contract CSV file.

        Parameters
        ----------
        filepath : Path
            Path to CSV file.
        category : Category
            Contract category.

        Returns
        -------
        ContractTimeseries | None
            Loaded contract data or None if loading fails.
        """
        try:
            df = pd.read_csv(filepath)

            if df.empty:
                logger.warning(f"Empty CSV file: {filepath}")
                return None

            contract_id = self._extract_contract_id(df, filepath)
            df = self._standardize_columns(df)
            df = self._parse_timestamps(df)
            df = self._set_index(df)
            df = self._add_identifiers(df, contract_id, category)
            frequency = self._detect_frequency(df)

            return ContractTimeseries(
                contract_id=contract_id,
                category=category,
                data=df,
                frequency=frequency,
            )

        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            return None

    def _extract_contract_id(self, df: pd.DataFrame, filepath: Path) -> ContractID:
        """Extract contract ID from data or filename."""
        if "contract_id" in df.columns and df["contract_id"].notna().any():
            return str(df["contract_id"].dropna().iloc[0])
        return filepath.stem

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names."""
        column_mapping = {
            "datetime": "timestamp",
        }
        df = df.rename(columns=column_mapping)
        return df

    def _parse_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse and convert timestamps, dropping invalid rows."""
        if "timestamp" not in df.columns:
            raise ValueError("No timestamp column found")

        df["timestamp"] = self.timestamp_parser.parse(df["timestamp"])

        # Drop rows with invalid timestamps (NaT)
        n_before = len(df)
        df = df.dropna(subset=["timestamp"])
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            logger.debug(f"Dropped {n_dropped} rows with invalid timestamps")

        return df

    def _set_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Set timestamp as index."""
        df = df.set_index("timestamp")
        df = df.sort_index()
        return df

    def _add_identifiers(
        self, df: pd.DataFrame, contract_id: ContractID, category: Category
    ) -> pd.DataFrame:
        """Add contract identifiers to DataFrame with canonical category."""
        df["contract_id"] = contract_id
        df["category"] = canonical_category(category)
        return df

    def _detect_frequency(self, df: pd.DataFrame) -> DataFrequency:
        """Detect data sampling frequency."""
        if len(df) < 2:
            return DataFrequency.MINUTE

        time_diffs = df.index.to_series().diff().dropna()
        if time_diffs.empty:
            return DataFrequency.MINUTE

        median_diff = time_diffs.median()
        seconds = median_diff.total_seconds()

        if seconds <= 1.5:
            return DataFrequency.SECOND
        elif seconds <= 90:
            return DataFrequency.MINUTE
        elif seconds <= 5400:
            return DataFrequency.HOUR
        else:
            return DataFrequency.DAY


class MetadataLoader:
    """
    Loads and parses contract metadata.

    Handles the metadata.csv file containing contract-level
    information and resolution rules.
    """

    def __init__(self, config: IRPConfig, validator: SchemaValidator | None = None):
        """
        Initialize metadata loader.

        Parameters
        ----------
        config : IRPConfig
            Platform configuration.
        validator : SchemaValidator | None
            Optional schema validator.
        """
        self.config = config
        self.validator = validator

    def load(self, filepath: Path | None = None) -> pd.DataFrame:
        """
        Load metadata CSV file.

        Parameters
        ----------
        filepath : Path | None
            Path to metadata file. Uses config default if None.

        Returns
        -------
        pd.DataFrame
            Metadata DataFrame indexed by contract_id.
        """
        if filepath is None:
            filepath = self.config.data.metadata_path

        if not filepath.exists():
            logger.warning(f"Metadata file not found: {filepath}")
            return pd.DataFrame()

        df = pd.read_csv(filepath)

        if self.validator:
            is_valid, missing = self.validator.validate_metadata(df)
            if not is_valid:
                logger.warning(f"Metadata schema mismatch. Missing: {missing}")

        df = self._parse_dates(df)
        df = self._standardize_categories(df)

        if "contract_id" in df.columns:
            df = df.set_index("contract_id")

        return df

    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse date columns."""
        date_cols = ["start_date", "end_date"]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
        return df

    def _standardize_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize category names."""
        if "pm_category" in df.columns:
            df["category"] = df["pm_category"]
        return df

    def get_contract_metadata(self, contract_id: ContractID) -> ContractMetadata | None:
        """
        Get metadata for specific contract.

        Parameters
        ----------
        contract_id : ContractID
            Contract identifier.

        Returns
        -------
        ContractMetadata | None
            Contract metadata or None if not found.
        """
        metadata_df = self.load()
        if contract_id not in metadata_df.index:
            return None

        row = metadata_df.loc[contract_id]
        row["contract_id"] = contract_id
        return ContractMetadata.from_series(row)


class KalshiDataLoader:
    """
    Main data loader for Kalshi prediction market data.

    Orchestrates loading of all contracts and metadata into
    a unified MarketDataset structure.
    """

    def __init__(self, config: IRPConfig | None = None):
        """
        Initialize Kalshi data loader.

        Parameters
        ----------
        config : IRPConfig | None
            Platform configuration. Uses global config if None.
        """
        self.config = config or get_config()
        self.validator = SchemaValidator(self.config.data.schema_path)
        self.timestamp_parser = TimestampParser(
            input_tz=self.config.timezone.input,
            output_tz=self.config.timezone.output,
        )
        self.contract_loader = ContractLoader(
            self.config, self.timestamp_parser, self.validator
        )
        self.metadata_loader = MetadataLoader(self.config, self.validator)

    def load_all(
        self,
        categories: list[Category] | None = None,
        min_observations: int | None = None,
    ) -> MarketDataset:
        """
        Load all contracts from configured data path.

        Parameters
        ----------
        categories : list[Category] | None
            Categories to load. Loads all if None.
        min_observations : int | None
            Minimum observations required. Uses config default if None.

        Returns
        -------
        MarketDataset
            Dataset containing all loaded contracts.
        """
        if categories is None:
            categories = self.config.categories

        if min_observations is None:
            min_observations = self.config.cleaning.min_observations

        logger.info(f"Loading contracts from {self.config.data.csv_path}")
        logger.info(f"Categories: {categories}")

        metadata = self.metadata_loader.load()
        dataset = MarketDataset(metadata=metadata)

        for category in categories:
            category_contracts = self._load_category(category)
            for contract in category_contracts:
                if contract.n_observations >= min_observations:
                    # Attach metadata if available
                    if contract.contract_id in metadata.index:
                        contract.metadata = ContractMetadata.from_series(
                            metadata.loc[contract.contract_id].copy().to_frame().T.iloc[0]
                        )
                        contract.metadata.contract_id = contract.contract_id

                    # Infer event time based on category (Section 2)
                    contract.event_time = infer_event_time(contract, contract.metadata)

                    dataset.add(contract)
                else:
                    logger.debug(
                        f"Skipping {contract.contract_id}: "
                        f"{contract.n_observations} < {min_observations} obs"
                    )

        logger.info(f"Loaded {len(dataset)} contracts")
        return dataset

    def _load_category(self, category: Category) -> Iterator[ContractTimeseries]:
        """
        Load all contracts for a category.

        Parameters
        ----------
        category : Category
            Category to load.

        Yields
        ------
        ContractTimeseries
            Loaded contract data.
        """
        category_path = self.config.data.csv_path / category

        if not category_path.exists():
            logger.warning(f"Category directory not found: {category_path}")
            return

        csv_files = list(category_path.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files in {category}")

        for filepath in csv_files:
            contract = self.contract_loader.load(filepath, category)
            if contract is not None:
                yield contract

    def load_contract(
        self, contract_id: ContractID, category: Category | None = None
    ) -> ContractTimeseries | None:
        """
        Load a specific contract by ID.

        Parameters
        ----------
        contract_id : ContractID
            Contract identifier.
        category : Category | None
            Category hint. Searches all categories if None.

        Returns
        -------
        ContractTimeseries | None
            Contract data or None if not found.
        """
        if category is not None:
            filepath = self.config.data.csv_path / category / f"{contract_id}.csv"
            if filepath.exists():
                return self.contract_loader.load(filepath, category)
            return None

        for cat in self.config.categories:
            filepath = self.config.data.csv_path / cat / f"{contract_id}.csv"
            if filepath.exists():
                return self.contract_loader.load(filepath, cat)

        logger.warning(f"Contract not found: {contract_id}")
        return None

    def get_available_contracts(self) -> pd.DataFrame:
        """
        Get list of available contracts without loading data.

        Returns
        -------
        pd.DataFrame
            DataFrame with contract_id, category, and file info.
        """
        records = []
        for category in self.config.categories:
            category_path = self.config.data.csv_path / category
            if not category_path.exists():
                continue

            for filepath in category_path.glob("*.csv"):
                records.append({
                    "contract_id": filepath.stem,
                    "category": category,
                    "filepath": str(filepath),
                    "file_size_kb": filepath.stat().st_size / 1024,
                })

        return pd.DataFrame(records)


def load_kalshi_data(
    config: IRPConfig | None = None,
    categories: list[Category] | None = None,
    min_observations: int | None = None,
) -> MarketDataset:
    """
    Convenience function to load Kalshi market data.

    Parameters
    ----------
    config : IRPConfig | None
        Platform configuration.
    categories : list[Category] | None
        Categories to load.
    min_observations : int | None
        Minimum observations filter.

    Returns
    -------
    MarketDataset
        Loaded market dataset.
    """
    loader = KalshiDataLoader(config)
    return loader.load_all(categories, min_observations)
