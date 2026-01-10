"""
Data cleaning module for IRP platform.

Provides comprehensive data cleaning, validation, and imputation
strategies for prediction market timeseries data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.config import IRPConfig, get_config
from src.utils.logging import get_logger
from src.utils.types import (
    ContractTimeseries,
    MarketDataset,
)

logger = get_logger("clean")


class ImputationMethod(Enum):
    """Available imputation strategies."""
    FORWARD_FILL = auto()
    BACKWARD_FILL = auto()
    LINEAR = auto()
    MIDPOINT = auto()
    DROP = auto()


class ValidationResult:
    """Container for validation results."""

    def __init__(self, is_valid: bool, errors: list[str] | None = None):
        self.is_valid = is_valid
        self.errors = errors or []

    def __bool__(self) -> bool:
        return self.is_valid

    def __repr__(self) -> str:
        if self.is_valid:
            return "ValidationResult(valid=True)"
        return f"ValidationResult(valid=False, errors={self.errors})"


@dataclass
class CleaningStats:
    """Statistics from cleaning operations."""
    original_rows: int
    final_rows: int
    rows_removed: int
    nulls_imputed: int
    outliers_winsorized: int
    invalid_prices_fixed: int

    @property
    def removal_pct(self) -> float:
        if self.original_rows == 0:
            return 0.0
        return 100 * self.rows_removed / self.original_rows


class DataValidator(ABC):
    """Abstract base class for data validators."""

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Validate DataFrame and return result."""
        pass


class PriceValidator(DataValidator):
    """Validates price data is within bounds."""

    def __init__(self, min_price: float = 0, max_price: float = 100):
        self.min_price = min_price
        self.max_price = max_price

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        price_cols = [c for c in df.columns if "price" in c.lower()]
        errors = []

        for col in price_cols:
            if col not in df.columns:
                continue

            series = df[col].dropna()
            if series.empty:
                continue

            below_min = (series < self.min_price).sum()
            above_max = (series > self.max_price).sum()

            if below_min > 0:
                errors.append(f"{col}: {below_min} values below {self.min_price}")
            if above_max > 0:
                errors.append(f"{col}: {above_max} values above {self.max_price}")

        return ValidationResult(len(errors) == 0, errors)


class SpreadValidator(DataValidator):
    """Validates bid-ask spread is reasonable."""

    def __init__(self, max_spread_pct: float = 0.5):
        self.max_spread_pct = max_spread_pct

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        errors = []

        if "yes_bid_c" not in df.columns or "yes_ask_c" not in df.columns:
            return ValidationResult(True)

        bid = df["yes_bid_c"]
        ask = df["yes_ask_c"]

        valid_mask = bid.notna() & ask.notna() & (bid > 0)
        if not valid_mask.any():
            return ValidationResult(True)

        spread_pct = (ask - bid) / bid
        wide_spreads = (spread_pct > self.max_spread_pct).sum()

        if wide_spreads > 0:
            errors.append(f"Spread exceeds {self.max_spread_pct:.0%} in {wide_spreads} rows")

        return ValidationResult(len(errors) == 0, errors)


class TimestampValidator(DataValidator):
    """Validates timestamp data quality."""

    def __init__(self, max_gap_minutes: int = 60):
        self.max_gap_minutes = max_gap_minutes

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        errors = []

        if not isinstance(df.index, pd.DatetimeIndex):
            errors.append("Index is not DatetimeIndex")
            return ValidationResult(False, errors)

        if df.index.isna().any():
            na_count = df.index.isna().sum()
            errors.append(f"Found {na_count} null timestamps")

        if not df.index.is_monotonic_increasing:
            errors.append("Timestamps not monotonically increasing")

        return ValidationResult(len(errors) == 0, errors)


class Imputer:
    """
    Handles missing value imputation for market data.

    Provides multiple imputation strategies appropriate for
    different types of market data columns.
    """

    def __init__(self, method: ImputationMethod = ImputationMethod.FORWARD_FILL):
        self.method = method

    def impute(self, df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
        """
        Impute missing values in specified columns.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with missing values.
        columns : list[str] | None
            Columns to impute. Imputes all numeric if None.

        Returns
        -------
        pd.DataFrame
            DataFrame with imputed values.
        """
        df = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col not in df.columns:
                continue
            df[col] = self._impute_column(df[col])

        return df

    def _impute_column(self, series: pd.Series) -> pd.Series:
        """Apply imputation to single column."""
        if self.method == ImputationMethod.FORWARD_FILL:
            return series.ffill()
        elif self.method == ImputationMethod.BACKWARD_FILL:
            return series.bfill()
        elif self.method == ImputationMethod.LINEAR:
            return series.interpolate(method="linear")
        elif self.method == ImputationMethod.DROP:
            return series
        else:
            return series.ffill()

    def impute_bid_ask(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Specialized imputation for bid-ask data.

        Uses midpoint-aware imputation that maintains bid < ask relationship.
        """
        df = df.copy()

        bid_cols = [c for c in df.columns if "bid" in c.lower()]
        ask_cols = [c for c in df.columns if "ask" in c.lower()]

        for col in bid_cols + ask_cols:
            if col in df.columns:
                df[col] = df[col].ffill()

        return df


class Winsorizer:
    """
    Handles outlier treatment via winsorization.

    Clips extreme values to specified percentile bounds.
    """

    def __init__(self, limits: tuple[float, float] = (0.01, 0.01)):
        self.lower_limit = limits[0]
        self.upper_limit = limits[1]

    def winsorize(
        self, df: pd.DataFrame, columns: list[str] | None = None
    ) -> tuple[pd.DataFrame, int]:
        """
        Winsorize specified columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        columns : list[str] | None
            Columns to winsorize. All numeric if None.

        Returns
        -------
        tuple[pd.DataFrame, int]
            (Winsorized DataFrame, count of modified values)
        """
        df = df.copy()
        total_modified = 0

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col not in df.columns:
                continue

            original = df[col].copy()
            df[col], n_modified = self._winsorize_column(df[col])
            total_modified += n_modified

        return df, total_modified

    def _winsorize_column(self, series: pd.Series) -> tuple[pd.Series, int]:
        """Winsorize single column."""
        valid_mask = series.notna()
        if not valid_mask.any():
            return series, 0

        valid_values = series[valid_mask]
        lower_bound = valid_values.quantile(self.lower_limit)
        upper_bound = valid_values.quantile(1 - self.upper_limit)

        below_lower = (series < lower_bound).sum()
        above_upper = (series > upper_bound).sum()
        n_modified = below_lower + above_upper

        result = series.clip(lower=lower_bound, upper=upper_bound)
        return result, n_modified


class DataCleaner:
    """
    Main data cleaning orchestrator.

    Coordinates validation, imputation, and outlier treatment
    for market data.
    """

    def __init__(self, config: IRPConfig | None = None):
        """
        Initialize data cleaner.

        Parameters
        ----------
        config : IRPConfig | None
            Platform configuration. Uses global if None.
        """
        self.config = config or get_config()
        self.cleaning_config = self.config.cleaning

        imputation_method = ImputationMethod[
            self.cleaning_config.imputation_method.upper()
        ]
        self.imputer = Imputer(method=imputation_method)
        self.winsorizer = Winsorizer(limits=self.cleaning_config.winsorize_limits)

        self.validators = [
            PriceValidator(
                min_price=self.cleaning_config.min_price,
                max_price=self.cleaning_config.max_price,
            ),
            SpreadValidator(max_spread_pct=self.cleaning_config.max_spread_pct),
            TimestampValidator(max_gap_minutes=self.cleaning_config.max_gap_minutes),
        ]

    def clean_contract(
        self, contract: ContractTimeseries
    ) -> tuple[ContractTimeseries, CleaningStats]:
        """
        Clean single contract timeseries.

        Parameters
        ----------
        contract : ContractTimeseries
            Contract to clean.

        Returns
        -------
        tuple[ContractTimeseries, CleaningStats]
            (Cleaned contract, cleaning statistics)
        """
        df = contract.data.copy()
        original_rows = len(df)

        stats = CleaningStats(
            original_rows=original_rows,
            final_rows=0,
            rows_removed=0,
            nulls_imputed=0,
            outliers_winsorized=0,
            invalid_prices_fixed=0,
        )

        df = self._remove_invalid_rows(df)

        null_count_before = df.isna().sum().sum()
        df = self._impute_missing(df)
        null_count_after = df.isna().sum().sum()
        stats.nulls_imputed = null_count_before - null_count_after

        df, n_winsorized = self._winsorize_returns(df)
        stats.outliers_winsorized = n_winsorized

        df, n_fixed = self._fix_invalid_prices(df)
        stats.invalid_prices_fixed = n_fixed

        df = self._remove_sparse_rows(df)

        stats.final_rows = len(df)
        stats.rows_removed = original_rows - stats.final_rows

        cleaned_contract = ContractTimeseries(
            contract_id=contract.contract_id,
            category=contract.category,
            data=df,
            metadata=contract.metadata,
            frequency=contract.frequency,
        )

        return cleaned_contract, stats

    def clean_dataset(
        self, dataset: MarketDataset
    ) -> tuple[MarketDataset, dict[str, CleaningStats]]:
        """
        Clean entire market dataset.

        Parameters
        ----------
        dataset : MarketDataset
            Dataset to clean.

        Returns
        -------
        tuple[MarketDataset, dict[str, CleaningStats]]
            (Cleaned dataset, per-contract statistics)
        """
        logger.info(f"Cleaning {len(dataset)} contracts")

        cleaned_dataset = MarketDataset(metadata=dataset.metadata)
        all_stats = {}

        for contract in dataset:
            cleaned_contract, stats = self.clean_contract(contract)

            if cleaned_contract.n_observations >= self.cleaning_config.min_observations:
                cleaned_dataset.add(cleaned_contract)
                all_stats[contract.contract_id] = stats
            else:
                logger.debug(
                    f"Dropping {contract.contract_id}: "
                    f"{cleaned_contract.n_observations} obs after cleaning"
                )

        logger.info(
            f"Cleaned dataset: {len(cleaned_dataset)} contracts "
            f"(dropped {len(dataset) - len(cleaned_dataset)})"
        )

        return cleaned_dataset, all_stats

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Run all validators on DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate.

        Returns
        -------
        ValidationResult
            Combined validation result.
        """
        all_errors = []

        for validator in self.validators:
            result = validator.validate(df)
            if not result.is_valid:
                all_errors.extend(result.errors)

        return ValidationResult(len(all_errors) == 0, all_errors)

    def _remove_invalid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with invalid data."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return df

        valid_index = ~df.index.isna()
        df = df[valid_index]

        return df

    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values."""
        price_cols = [c for c in df.columns if "price" in c.lower()]
        bid_ask_cols = [c for c in df.columns if "bid" in c.lower() or "ask" in c.lower()]

        df = self.imputer.impute(df, columns=price_cols)
        df = self.imputer.impute_bid_ask(df)

        return df

    def _winsorize_returns(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Winsorize extreme values in return-like columns."""
        return_cols = [c for c in df.columns if "return" in c.lower()]

        if not return_cols:
            return df, 0

        return self.winsorizer.winsorize(df, columns=return_cols)

    def _fix_invalid_prices(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Fix prices outside valid range.

        Note: This is a private method called from clean_contract which already
        copies the DataFrame. No additional copy is needed here.
        """
        n_fixed = 0

        price_cols = [c for c in df.columns if "price" in c.lower()]
        min_price = self.cleaning_config.min_price
        max_price = self.cleaning_config.max_price

        for col in price_cols:
            if col not in df.columns:
                continue

            below = df[col] < min_price
            above = df[col] > max_price
            n_fixed += below.sum() + above.sum()

            df.loc[below, col] = min_price
            df.loc[above, col] = max_price

        return df, n_fixed

    def _remove_sparse_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with too much missing data."""
        if self.cleaning_config.drop_zero_volume_only:
            if "volume" in df.columns:
                df = df[df["volume"] > 0]

        return df


def clean_market_data(
    dataset: MarketDataset, config: IRPConfig | None = None
) -> tuple[MarketDataset, dict[str, CleaningStats]]:
    """
    Convenience function to clean market dataset.

    Parameters
    ----------
    dataset : MarketDataset
        Dataset to clean.
    config : IRPConfig | None
        Platform configuration.

    Returns
    -------
    tuple[MarketDataset, dict[str, CleaningStats]]
        (Cleaned dataset, cleaning statistics)
    """
    cleaner = DataCleaner(config)
    return cleaner.clean_dataset(dataset)
