"""
Amplitude normalization utilities for cross-contract analysis.

Provides z-score, percentile, and min-max normalization for
standardizing metrics across contracts with different scales.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.config import IRPConfig, get_config
from src.utils.logging import get_logger

logger = get_logger("normalization")


class NormMethod(Enum):
    """Available normalization methods."""
    ZSCORE = auto()
    PERCENTILE = auto()
    MINMAX = auto()
    ROBUST = auto()
    NONE = auto()


def zscore_normalize(
    series: pd.Series,
    window: int | None = None,
) -> pd.Series:
    """
    Z-score normalization.

    Parameters
    ----------
    series : pd.Series
        Series to normalize.
    window : int | None
        Rolling window for local z-score. Global if None.

    Returns
    -------
    pd.Series
        Z-score normalized series.
    """
    if window is not None:
        rolling_mean = series.rolling(window=window, min_periods=1).mean()
        rolling_std = series.rolling(window=window, min_periods=1).std()
        return (series - rolling_mean) / rolling_std.replace(0, np.nan)
    else:
        mean = series.mean()
        std = series.std()
        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=series.index)
        return (series - mean) / std


def percentile_normalize(
    series: pd.Series,
    lower: float = 0.0,
    upper: float = 100.0,
) -> pd.Series:
    """
    Percentile rank normalization.

    Parameters
    ----------
    series : pd.Series
        Series to normalize.
    lower : float
        Lower percentile bound.
    upper : float
        Upper percentile bound.

    Returns
    -------
    pd.Series
        Percentile normalized series (0-1 scale).
    """
    valid = series.dropna()
    if len(valid) == 0:
        return pd.Series(np.nan, index=series.index)

    ranks = series.rank(pct=True, na_option="keep")

    if lower > 0 or upper < 100:
        lower_val = series.quantile(lower / 100)
        upper_val = series.quantile(upper / 100)
        series_clipped = series.clip(lower=lower_val, upper=upper_val)
        ranks = series_clipped.rank(pct=True, na_option="keep")

    return ranks


def minmax_normalize(
    series: pd.Series,
    feature_range: tuple[float, float] = (0.0, 1.0),
    clip_percentiles: tuple[float, float] | None = None,
) -> pd.Series:
    """
    Min-max normalization.

    Parameters
    ----------
    series : pd.Series
        Series to normalize.
    feature_range : tuple[float, float]
        Target range (min, max).
    clip_percentiles : tuple[float, float] | None
        Clip to percentile bounds before scaling.

    Returns
    -------
    pd.Series
        Min-max normalized series.
    """
    if clip_percentiles is not None:
        lower = series.quantile(clip_percentiles[0] / 100)
        upper = series.quantile(clip_percentiles[1] / 100)
        series = series.clip(lower=lower, upper=upper)

    min_val = series.min()
    max_val = series.max()

    if min_val == max_val:
        return pd.Series((feature_range[0] + feature_range[1]) / 2, index=series.index)

    scaled = (series - min_val) / (max_val - min_val)
    return scaled * (feature_range[1] - feature_range[0]) + feature_range[0]


def robust_normalize(
    series: pd.Series,
    quantile_range: tuple[float, float] = (25.0, 75.0),
) -> pd.Series:
    """
    Robust normalization using median and IQR.

    Parameters
    ----------
    series : pd.Series
        Series to normalize.
    quantile_range : tuple[float, float]
        Quantile range for IQR calculation.

    Returns
    -------
    pd.Series
        Robust normalized series.
    """
    median = series.median()
    q_low = series.quantile(quantile_range[0] / 100)
    q_high = series.quantile(quantile_range[1] / 100)
    iqr = q_high - q_low

    if iqr == 0:
        return pd.Series(0.0, index=series.index)

    return (series - median) / iqr


def normalize_series(
    series: pd.Series,
    method: str = "zscore",
    **kwargs,
) -> pd.Series:
    """
    Normalize series using specified method.

    Parameters
    ----------
    series : pd.Series
        Series to normalize.
    method : str
        Normalization method: "zscore", "percentile", "minmax", "robust", "none".
    **kwargs
        Additional arguments passed to normalization function.

    Returns
    -------
    pd.Series
        Normalized series.
    """
    method = method.lower()

    if method == "zscore":
        return zscore_normalize(series, **kwargs)
    elif method == "percentile":
        return percentile_normalize(series, **kwargs)
    elif method == "minmax":
        return minmax_normalize(series, **kwargs)
    elif method == "robust":
        return robust_normalize(series, **kwargs)
    elif method == "none":
        return series.copy()
    else:
        logger.warning(f"Unknown normalization method: {method}, using zscore")
        return zscore_normalize(series)


def normalize_dataframe(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: str | dict[str, str] = "zscore",
    config: IRPConfig | None = None,
    suffix: str = "_norm",
) -> pd.DataFrame:
    """
    Normalize multiple columns in DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to normalize.
    columns : list[str] | None
        Columns to normalize. All numeric if None.
    method : str | dict[str, str]
        Normalization method or dict mapping column to method.
    config : IRPConfig | None
        Configuration with normalization settings.
    suffix : str
        Suffix for normalized column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized columns added.
    """
    df = df.copy()

    if config is None:
        config = get_config()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col not in df.columns:
            continue

        if isinstance(method, dict):
            col_method = method.get(col, config.normalization.default)
        elif isinstance(method, str):
            col_method = method
        else:
            col_method = _get_config_method(col, config)

        norm_col = f"{col}{suffix}"
        df[norm_col] = normalize_series(df[col], method=col_method)

    return df


def _get_config_method(column: str, config: IRPConfig) -> str:
    """Get normalization method from config based on column name."""
    col_lower = column.lower()

    if "spread" in col_lower:
        return config.normalization.spread
    elif "volume" in col_lower:
        return config.normalization.volume
    elif "depth" in col_lower or "book" in col_lower:
        return config.normalization.depth
    elif "volatility" in col_lower or "vol_" in col_lower:
        return config.normalization.volatility
    elif "price" in col_lower:
        return config.normalization.price
    else:
        return config.normalization.default


def normalize_cross_contract(
    dfs: dict[str, pd.DataFrame],
    column: str,
    method: str = "zscore",
    global_stats: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Normalize column across multiple contracts.

    Parameters
    ----------
    dfs : dict[str, pd.DataFrame]
        Dictionary mapping contract_id to DataFrame.
    column : str
        Column to normalize.
    method : str
        Normalization method.
    global_stats : bool
        Use global statistics across all contracts.

    Returns
    -------
    dict[str, pd.DataFrame]
        DataFrames with normalized column.
    """
    if global_stats:
        all_values = pd.concat([df[column] for df in dfs.values() if column in df.columns])

        if method == "zscore":
            global_mean = all_values.mean()
            global_std = all_values.std()
        elif method == "minmax":
            global_min = all_values.min()
            global_max = all_values.max()
        elif method == "percentile":
            pass

    result = {}
    for contract_id, df in dfs.items():
        df = df.copy()
        if column not in df.columns:
            result[contract_id] = df
            continue

        norm_col = f"{column}_norm"

        if global_stats:
            if method == "zscore":
                df[norm_col] = (df[column] - global_mean) / global_std
            elif method == "minmax":
                if global_max != global_min:
                    df[norm_col] = (df[column] - global_min) / (global_max - global_min)
                else:
                    df[norm_col] = 0.5
            elif method == "percentile":
                df[norm_col] = df[column].apply(
                    lambda x: (all_values < x).sum() / len(all_values)
                )
        else:
            df[norm_col] = normalize_series(df[column], method=method)

        result[contract_id] = df

    return result


class Normalizer:
    """
    Stateful normalizer for consistent cross-dataset normalization.

    Fits on training data and transforms new data using same parameters.
    """

    def __init__(self, method: str = "zscore"):
        """
        Initialize normalizer.

        Parameters
        ----------
        method : str
            Normalization method.
        """
        self.method = method
        self.params: dict[str, dict] = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame, columns: list[str] | None = None) -> "Normalizer":
        """
        Fit normalizer on data.

        Parameters
        ----------
        df : pd.DataFrame
            Training data.
        columns : list[str] | None
            Columns to fit. All numeric if None.

        Returns
        -------
        Normalizer
            Self for chaining.
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col not in df.columns:
                continue

            series = df[col].dropna()

            if self.method == "zscore":
                self.params[col] = {
                    "mean": series.mean(),
                    "std": series.std(),
                }
            elif self.method == "minmax":
                self.params[col] = {
                    "min": series.min(),
                    "max": series.max(),
                }
            elif self.method == "robust":
                self.params[col] = {
                    "median": series.median(),
                    "q25": series.quantile(0.25),
                    "q75": series.quantile(0.75),
                }

        self._fitted = True
        return self

    def transform(
        self, df: pd.DataFrame, columns: list[str] | None = None, suffix: str = "_norm"
    ) -> pd.DataFrame:
        """
        Transform data using fitted parameters.

        Parameters
        ----------
        df : pd.DataFrame
            Data to transform.
        columns : list[str] | None
            Columns to transform.
        suffix : str
            Suffix for normalized columns.

        Returns
        -------
        pd.DataFrame
            Transformed data.
        """
        if not self._fitted:
            raise RuntimeError("Normalizer must be fitted before transform")

        df = df.copy()

        if columns is None:
            columns = list(self.params.keys())

        for col in columns:
            if col not in df.columns or col not in self.params:
                continue

            params = self.params[col]
            norm_col = f"{col}{suffix}"

            if self.method == "zscore":
                std = params["std"]
                if std == 0 or pd.isna(std):
                    df[norm_col] = 0.0
                else:
                    df[norm_col] = (df[col] - params["mean"]) / std

            elif self.method == "minmax":
                range_val = params["max"] - params["min"]
                if range_val == 0:
                    df[norm_col] = 0.5
                else:
                    df[norm_col] = (df[col] - params["min"]) / range_val

            elif self.method == "robust":
                iqr = params["q75"] - params["q25"]
                if iqr == 0:
                    df[norm_col] = 0.0
                else:
                    df[norm_col] = (df[col] - params["median"]) / iqr

        return df

    def fit_transform(
        self, df: pd.DataFrame, columns: list[str] | None = None, suffix: str = "_norm"
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df, columns)
        return self.transform(df, columns, suffix)
