"""
Time binning utilities for cross-contract analysis.

Provides normalized time axes for comparing contracts with different
lifespans and event times. This module is the SINGLE SOURCE OF TRUTH
for all lifecycle computations across the ELTR platform.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Literal

import numpy as np
import pandas as pd

from src.utils.config import ELTRConfig, get_config
from src.utils.logging import get_logger

logger = get_logger("time_binning")


class TimeBinMode(Enum):
    """Available time binning modes."""
    LIFECYCLE = auto()
    TTS_HOURS = auto()
    EVENT_ALIGNED = auto()


# Nonlinear lifecycle bin edges per Section 10
NONLINEAR_BIN_EDGES = [0.0, 0.05, 0.20, 0.80, 0.95, 1.0]
NONLINEAR_BIN_LABELS = [0, 1, 2, 3, 4]


def compute_lifecycle_features(
    df: pd.DataFrame,
    listing_time: pd.Timestamp | None = None,
    settlement_time: pd.Timestamp | None = None,
    event_time: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Compute contract lifecycle normalization features.

    This is the SINGLE SOURCE OF TRUTH for lifecycle computation.
    All other modules must use this function.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex.
    listing_time : pd.Timestamp | None
        Contract listing time. Infers from data if None.
    settlement_time : pd.Timestamp | None
        Contract settlement time. Infers from data if None.
    event_time : pd.Timestamp | None
        Event time (for sports/econ events). Used as settlement if provided.

    Returns
    -------
    pd.DataFrame
        DataFrame with lifecycle features added:
        - tsl_hours: Time since listing in hours
        - tts_hours: Time to settlement in hours
        - lifecycle_ratio: Normalized 0→1 lifecycle position
        - lifecycle_phase: Categorical phase label
    """
    df = df.copy()

    if listing_time is None:
        listing_time = df.index.min()

    # Event time takes precedence over settlement time
    if settlement_time is None:
        if event_time is not None:
            settlement_time = event_time
        else:
            settlement_time = df.index.max()

    # Handle timezone conversion
    if listing_time.tzinfo is None and df.index.tzinfo is not None:
        listing_time = listing_time.tz_localize(df.index.tzinfo)
    if settlement_time.tzinfo is None and df.index.tzinfo is not None:
        settlement_time = settlement_time.tz_localize(df.index.tzinfo)

    total_duration = (settlement_time - listing_time).total_seconds() / 3600

    df["tsl_hours"] = (df.index - listing_time).total_seconds() / 3600
    df["tts_hours"] = (settlement_time - df.index).total_seconds() / 3600

    if total_duration > 0:
        df["lifecycle_ratio"] = df["tsl_hours"] / total_duration
    else:
        df["lifecycle_ratio"] = 0.0

    df["lifecycle_ratio"] = df["lifecycle_ratio"].clip(0, 1)

    # Add lifecycle phase using nonlinear bins
    df["lifecycle_phase"] = pd.cut(
        df["lifecycle_ratio"],
        bins=NONLINEAR_BIN_EDGES,
        labels=["early", "ramp_up", "middle", "late", "resolution"],
        include_lowest=True,
    )

    return df


def make_lifecycle_bins(
    df: pd.DataFrame,
    n_bins: int = 50,
    lifecycle_col: str = "lifecycle_ratio",
    nonlinear: bool = False,
) -> pd.Series:
    """
    Assign observations to lifecycle bins.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with lifecycle_ratio column.
    n_bins : int
        Number of bins across lifecycle (ignored if nonlinear=True).
    lifecycle_col : str
        Name of lifecycle column.
    nonlinear : bool
        Use nonlinear bin edges per Section 10.

    Returns
    -------
    pd.Series
        Integer bin labels (not float).
    """
    if lifecycle_col not in df.columns:
        raise ValueError(f"Column {lifecycle_col} not found")

    if nonlinear:
        # Use predefined nonlinear edges
        bins = pd.cut(
            df[lifecycle_col],
            bins=NONLINEAR_BIN_EDGES,
            labels=NONLINEAR_BIN_LABELS,
            include_lowest=True,
        )
        return bins.astype(int)
    else:
        # Linear bins
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bins = pd.cut(
            df[lifecycle_col],
            bins=bin_edges,
            labels=range(n_bins),
            include_lowest=True,
        )
        return bins.astype(int)


def make_lifecycle_bins_df(
    df: pd.DataFrame,
    n_bins: int = 50,
    lifecycle_col: str = "lifecycle_ratio",
    nonlinear: bool = False,
) -> pd.DataFrame:
    """
    Assign observations to lifecycle bins, returning full DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with lifecycle_ratio column.
    n_bins : int
        Number of bins across lifecycle.
    lifecycle_col : str
        Name of lifecycle column.
    nonlinear : bool
        Use nonlinear bin edges.

    Returns
    -------
    pd.DataFrame
        DataFrame with lifecycle_bin column added.
    """
    df = df.copy()
    df["lifecycle_bin"] = make_lifecycle_bins(df, n_bins, lifecycle_col, nonlinear)

    if nonlinear:
        # Compute bin centers for nonlinear case
        edges = np.array(NONLINEAR_BIN_EDGES)
        centers = (edges[:-1] + edges[1:]) / 2
        df["lifecycle_bin_center"] = df["lifecycle_bin"].map(
            {i: centers[i] for i in range(len(NONLINEAR_BIN_LABELS))}
        )
    else:
        df["lifecycle_bin_center"] = df["lifecycle_bin"] / n_bins + 0.5 / n_bins

    return df


def make_tts_bins(
    df: pd.DataFrame,
    bin_hours: float = 1.0,
    max_hours: float = 48.0,
    tts_col: str = "tts_hours",
) -> pd.DataFrame:
    """
    Assign observations to time-to-settlement bins.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with tts_hours column.
    bin_hours : float
        Width of each bin in hours.
    max_hours : float
        Maximum hours before settlement to include.
    tts_col : str
        Name of TTS column.

    Returns
    -------
    pd.DataFrame
        DataFrame with tts_bin column added.
    """
    df = df.copy()

    if tts_col not in df.columns:
        raise ValueError(f"Column {tts_col} not found")

    n_bins = int(max_hours / bin_hours)
    bin_edges = np.linspace(0, max_hours, n_bins + 1)

    tts_clipped = df[tts_col].clip(0, max_hours)
    df["tts_bin"] = pd.cut(
        tts_clipped,
        bins=bin_edges,
        labels=range(n_bins),
        include_lowest=True,
    )
    df["tts_bin"] = df["tts_bin"].astype(int)
    df["tts_bin_hours"] = df["tts_bin"] * bin_hours + bin_hours / 2

    return df


def make_event_aligned_bins(
    df: pd.DataFrame,
    event_time: pd.Timestamp,
    bin_minutes: float = 5.0,
    hours_pre: float = 24.0,
    hours_post: float = 4.0,
) -> pd.DataFrame:
    """
    Assign observations to event-aligned time bins.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex.
    event_time : pd.Timestamp
        Event timestamp for alignment.
    bin_minutes : float
        Width of each bin in minutes.
    hours_pre : float
        Hours before event to include.
    hours_post : float
        Hours after event to include.

    Returns
    -------
    pd.DataFrame
        DataFrame with event_bin column added.
    """
    df = df.copy()

    if event_time.tzinfo is None and df.index.tzinfo is not None:
        event_time = event_time.tz_localize(df.index.tzinfo)

    df["minutes_to_event"] = (event_time - df.index).total_seconds() / 60

    total_minutes = (hours_pre + hours_post) * 60
    n_bins = int(total_minutes / bin_minutes)

    minutes_from_start = df["minutes_to_event"] + hours_post * 60
    minutes_clipped = minutes_from_start.clip(0, total_minutes)

    bin_edges = np.linspace(0, total_minutes, n_bins + 1)
    df["event_bin"] = pd.cut(
        minutes_clipped,
        bins=bin_edges,
        labels=range(n_bins),
        include_lowest=True,
    )
    df["event_bin"] = df["event_bin"].astype(int)

    df["event_bin_minutes"] = -hours_post * 60 + df["event_bin"] * bin_minutes + bin_minutes / 2

    return df


def make_time_bins(
    df: pd.DataFrame,
    mode: str = "lifecycle",
    n_bins: int = 50,
    event_time: pd.Timestamp | None = None,
    config: ELTRConfig | None = None,
    nonlinear: bool = False,
) -> pd.DataFrame:
    """
    Universal time binning function.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to bin.
    mode : str
        Binning mode: "lifecycle", "tts_hours", or "event_aligned".
    n_bins : int
        Number of bins.
    event_time : pd.Timestamp | None
        Event time for event_aligned mode.
    config : ELTRConfig | None
        Configuration object.
    nonlinear : bool
        Use nonlinear bin edges (lifecycle mode only).

    Returns
    -------
    pd.DataFrame
        DataFrame with time bins added.
    """
    if config is None:
        config = get_config()

    if "lifecycle_ratio" not in df.columns:
        df = compute_lifecycle_features(df, event_time=event_time)

    if mode == "lifecycle":
        return make_lifecycle_bins_df(df, n_bins=n_bins, nonlinear=nonlinear)

    elif mode == "tts_hours":
        return make_tts_bins(
            df,
            bin_hours=1.0,
            max_hours=config.time_binning.event_window_hours_pre,
        )

    elif mode == "event_aligned":
        if event_time is None:
            event_time = df.index.max()
        return make_event_aligned_bins(
            df,
            event_time=event_time,
            hours_pre=config.time_binning.event_window_hours_pre,
            hours_post=config.time_binning.event_window_hours_post,
        )

    else:
        raise ValueError(f"Unknown binning mode: {mode}")


def aggregate_by_bins(
    df: pd.DataFrame,
    bin_col: str,
    agg_dict: dict[str, str | list[str]],
) -> pd.DataFrame:
    """
    Aggregate values within time bins.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with bin assignments.
    bin_col : str
        Column containing bin assignments.
    agg_dict : dict[str, str | list[str]]
        Mapping from column to aggregation function(s).

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame indexed by bin.
    """
    agg_spec = {}
    for col, funcs in agg_dict.items():
        if col in df.columns:
            if isinstance(funcs, str):
                agg_spec[col] = [funcs]
            else:
                agg_spec[col] = funcs

    if not agg_spec:
        return pd.DataFrame()

    result = df.groupby(bin_col).agg(agg_spec)
    result.columns = ["_".join(col).strip() for col in result.columns.values]

    return result


def compute_binned_trajectory(
    df: pd.DataFrame,
    value_col: str,
    mode: str = "lifecycle",
    n_bins: int = 50,
    percentiles: list[float] | None = None,
    smooth_window: int = 3,
) -> pd.DataFrame:
    """
    Compute trajectory with percentile bands across bins.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to process.
    value_col : str
        Column for trajectory values.
    mode : str
        Time binning mode.
    n_bins : int
        Number of bins.
    percentiles : list[float] | None
        Percentiles for bands. Defaults to [10, 25, 50, 75, 90].
    smooth_window : int
        Rolling median smoothing window (Section 14).

    Returns
    -------
    pd.DataFrame
        Trajectory with percentile columns.
    """
    if percentiles is None:
        percentiles = [10, 25, 50, 75, 90]

    df = make_time_bins(df, mode=mode, n_bins=n_bins)

    if mode == "lifecycle":
        bin_col = "lifecycle_bin"
        center_col = "lifecycle_bin_center"
    elif mode == "tts_hours":
        bin_col = "tts_bin"
        center_col = "tts_bin_hours"
    else:
        bin_col = "event_bin"
        center_col = "event_bin_minutes"

    def compute_percentiles(group):
        valid = group[value_col].dropna()
        if len(valid) == 0:
            return pd.Series({f"p{int(p)}": np.nan for p in percentiles})
        return pd.Series({
            f"p{int(p)}": valid.quantile(p / 100)
            for p in percentiles
        })

    result = df.groupby(bin_col).apply(compute_percentiles, include_groups=False)
    result["count"] = df.groupby(bin_col)[value_col].count()

    centers = df.groupby(bin_col)[center_col].first()
    result = result.join(centers)

    # Apply rolling median smoothing per Section 14
    if smooth_window > 1:
        for col in result.columns:
            if col.startswith("p") and col != "count":
                result[col] = result[col].rolling(
                    window=smooth_window, min_periods=1, center=True
                ).median()

    return result.reset_index()


def resample_to_1min(
    df: pd.DataFrame,
    volume_col: str = "volume",
    bid_col: str = "yes_bid_c",
    ask_col: str = "yes_ask_c",
    price_col: str = "price_c",
) -> pd.DataFrame:
    """
    Resample contract data to fixed 1-minute intervals (Section 3).

    This prevents false volatility bursts from irregular sampling.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex.
    volume_col : str
        Volume column name.
    bid_col : str
        Bid price column name.
    ask_col : str
        Ask price column name.
    price_col : str
        Trade price column name.

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame with regular 1-minute intervals.
    """
    df = df.copy()

    # Aggregation rules
    agg_dict = {}

    # Volume: sum within minute
    if volume_col in df.columns:
        agg_dict[volume_col] = "sum"

    # Prices: last value (forward-fill later)
    for col in [bid_col, ask_col, price_col]:
        if col in df.columns:
            agg_dict[col] = "last"

    # Trade count: sum
    if "trade_count" in df.columns:
        agg_dict["trade_count"] = "sum"

    # OHLC prices
    for suffix in ["_o", "_h", "_l", "_c"]:
        col = f"price{suffix}"
        if col in df.columns:
            if suffix == "_o":
                agg_dict[col] = "first"
            elif suffix == "_h":
                agg_dict[col] = "max"
            elif suffix == "_l":
                agg_dict[col] = "min"
            else:
                agg_dict[col] = "last"

    # Bid/ask OHLC
    for prefix in ["yes_bid", "yes_ask"]:
        for suffix in ["_h", "_l", "_c"]:
            col = f"{prefix}{suffix}"
            if col in df.columns:
                if suffix == "_h":
                    agg_dict[col] = "max"
                elif suffix == "_l":
                    agg_dict[col] = "min"
                else:
                    agg_dict[col] = "last"

    resampled = df.resample("1min").agg(agg_dict)

    # Forward-fill prices, leave volume as 0 for missing minutes
    price_cols = [c for c in resampled.columns if "bid" in c or "ask" in c or "price" in c]
    for col in price_cols:
        resampled[col] = resampled[col].ffill()

    # Fill missing volume with 0
    if volume_col in resampled.columns:
        resampled[volume_col] = resampled[volume_col].fillna(0)
    if "trade_count" in resampled.columns:
        resampled["trade_count"] = resampled["trade_count"].fillna(0)

    return resampled


def detect_gaps(
    df: pd.DataFrame,
    gap_threshold_minutes: float = 10.0,
) -> pd.Series:
    """
    Detect time gaps in data that exceed threshold.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex.
    gap_threshold_minutes : float
        Minimum gap size to flag.

    Returns
    -------
    pd.Series
        Boolean series where True indicates gap after that row.
    """
    if len(df) < 2:
        return pd.Series(False, index=df.index)

    time_diffs = df.index.to_series().diff().dt.total_seconds() / 60
    return time_diffs > gap_threshold_minutes
