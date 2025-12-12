"""
Contract Lifecycle Normalization.

This module is the SINGLE SOURCE OF TRUTH for all lifecycle computations.
All other modules must use these functions for lifecycle normalization.

Definitions:
    Time Since Listing (TSL) = current_time - listing_time
        AUTHOR MUST DEFINE FORMALLY: units (hours), timezone handling,
        listing time source

    Time To Settlement (TTS) = settlement_time - current_time
        AUTHOR MUST DEFINE FORMALLY: units (hours), handling of negative values,
        settlement time source (event time vs contract settlement)

    Lifecycle Ratio = TSL / (TSL + TTS) = TSL / total_duration
        Range: [0, 1] where 0 = listing, 1 = settlement
        AUTHOR MUST DEFINE FORMALLY: edge cases at boundaries,
        clipping behavior, interpretation

    Lifecycle Phases (nonlinear bins):
        EARLY: [0.00, 0.05)
        RAMP_UP: [0.05, 0.20)
        MIDDLE: [0.20, 0.80)
        LATE: [0.80, 0.95)
        RESOLUTION: [0.95, 1.00]
        AUTHOR MUST DEFINE FORMALLY: bin edge justification,
        phase naming rationale, empirical basis

    Binned Trajectory:
        Aggregation of metric values within lifecycle bins
        Percentile bands: p10, p25, p50, p75, p90
        AUTHOR MUST DEFINE FORMALLY: aggregation method,
        smoothing window (3), percentile selection

References:
    - Section 10: Nonlinear lifecycle bin edges
    - Section 14: Rolling median smoothing for trajectories
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger("microstructure.lifecycle")


# Constants
# Moved from: src/utils/time_binning.py

# Nonlinear lifecycle bin edges per Section 10
LIFECYCLE_BIN_EDGES = [0.0, 0.05, 0.20, 0.80, 0.95, 1.0]
LIFECYCLE_BIN_LABELS = [0, 1, 2, 3, 4]
LIFECYCLE_PHASE_NAMES = ["early", "ramp_up", "middle", "late", "resolution"]


# Lifecycle feature computation
# Moved from: src/utils/time_binning.py

def compute_lifecycle_features(
    df: pd.DataFrame,
    listing_time: pd.Timestamp | datetime | None = None,
    settlement_time: pd.Timestamp | datetime | None = None,
    event_time: pd.Timestamp | datetime | None = None,
) -> pd.DataFrame:
    """
    Compute contract lifecycle normalization features.

    This is the SINGLE SOURCE OF TRUTH for lifecycle computation.
    All other modules must use this function.

    Definition:
        tsl_hours = (current_time - listing_time) / 3600
        tts_hours = (settlement_time - current_time) / 3600
        lifecycle_ratio = tsl_hours / total_duration, clipped to [0, 1]
        lifecycle_phase = categorical phase from nonlinear bins
        AUTHOR MUST DEFINE FORMALLY: timezone handling, inference rules
        when times are not provided

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex.
    listing_time : pd.Timestamp | datetime | None
        Contract listing time. Infers from data if None.
    settlement_time : pd.Timestamp | datetime | None
        Contract settlement time. Infers from data if None.
    event_time : pd.Timestamp | datetime | None
        Event time (for sports/econ events). Used as settlement if provided.

    Returns
    -------
    pd.DataFrame
        DataFrame with lifecycle features:
        - tsl_hours: Time since listing in hours
        - tts_hours: Time to settlement in hours
        - lifecycle_ratio: Normalized 0→1 lifecycle position
        - lifecycle_phase: Categorical phase label

    Moved from
    ----------
    src/utils/time_binning.py:35-105 (compute_lifecycle_features)
    """
    df = df.copy()

    # Convert datetime to Timestamp
    if isinstance(listing_time, datetime):
        listing_time = pd.Timestamp(listing_time)
    if isinstance(settlement_time, datetime):
        settlement_time = pd.Timestamp(settlement_time)
    if isinstance(event_time, datetime):
        event_time = pd.Timestamp(event_time)

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
        bins=LIFECYCLE_BIN_EDGES,
        labels=LIFECYCLE_PHASE_NAMES,
        include_lowest=True,
    )

    return df


# Lifecycle binning
# Moved from: src/utils/time_binning.py

def make_lifecycle_bins(
    df: pd.DataFrame,
    n_bins: int = 50,
    lifecycle_col: str = "lifecycle_ratio",
    nonlinear: bool = False,
) -> pd.Series:
    """
    Assign observations to lifecycle bins.

    Definition:
        Linear bins: equal-width bins from 0 to 1
        Nonlinear bins: predefined edges [0.0, 0.05, 0.20, 0.80, 0.95, 1.0]
        AUTHOR MUST DEFINE FORMALLY: bin selection rationale,
        nonlinear edge justification

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with lifecycle_ratio column.
    n_bins : int
        Number of bins (ignored if nonlinear=True).
    lifecycle_col : str
        Name of lifecycle column.
    nonlinear : bool
        Use nonlinear bin edges per Section 10.

    Returns
    -------
    pd.Series
        Integer bin labels.

    Moved from
    ----------
    src/utils/time_binning.py:108-154 (make_lifecycle_bins)
    """
    if lifecycle_col not in df.columns:
        raise ValueError(f"Column {lifecycle_col} not found")

    if nonlinear:
        bins = pd.cut(
            df[lifecycle_col],
            bins=LIFECYCLE_BIN_EDGES,
            labels=LIFECYCLE_BIN_LABELS,
            include_lowest=True,
        )
        return bins.astype(int)
    else:
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bins = pd.cut(
            df[lifecycle_col],
            bins=bin_edges,
            labels=range(n_bins),
            include_lowest=True,
        )
        return bins.astype(int)


def assign_lifecycle_bins(
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
        Number of bins.
    lifecycle_col : str
        Name of lifecycle column.
    nonlinear : bool
        Use nonlinear bin edges.

    Returns
    -------
    pd.DataFrame
        DataFrame with lifecycle_bin and lifecycle_bin_center columns.

    Moved from
    ----------
    src/utils/time_binning.py:157-195 (make_lifecycle_bins_df)
    """
    df = df.copy()
    df["lifecycle_bin"] = make_lifecycle_bins(df, n_bins, lifecycle_col, nonlinear)

    if nonlinear:
        edges = np.array(LIFECYCLE_BIN_EDGES)
        centers = (edges[:-1] + edges[1:]) / 2
        df["lifecycle_bin_center"] = df["lifecycle_bin"].map(
            {i: centers[i] for i in range(len(LIFECYCLE_BIN_LABELS))}
        )
    else:
        df["lifecycle_bin_center"] = df["lifecycle_bin"] / n_bins + 0.5 / n_bins

    return df


# Binned trajectory computation
# Moved from: src/utils/time_binning.py

def compute_binned_trajectory(
    df: pd.DataFrame,
    value_col: str,
    n_bins: int = 50,
    percentiles: list[float] | None = None,
    smooth_window: int = 3,
    nonlinear: bool = False,
) -> pd.DataFrame:
    """
    Compute trajectory with percentile bands across lifecycle bins.

    Definition:
        Aggregates values within lifecycle bins.
        Computes percentile distributions: p10, p25, p50, p75, p90
        Applies rolling median smoothing (Section 14)
        AUTHOR MUST DEFINE FORMALLY: aggregation method,
        smoothing rationale, percentile selection

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to process.
    value_col : str
        Column for trajectory values.
    n_bins : int
        Number of bins.
    percentiles : list[float] | None
        Percentiles for bands. Defaults to [10, 25, 50, 75, 90].
    smooth_window : int
        Rolling median smoothing window (Section 14).
    nonlinear : bool
        Use nonlinear bin edges.

    Returns
    -------
    pd.DataFrame
        Trajectory with percentile columns.

    Moved from
    ----------
    src/utils/time_binning.py:399-468 (compute_binned_trajectory)
    """
    if percentiles is None:
        percentiles = [10, 25, 50, 75, 90]

    df = assign_lifecycle_bins(df, n_bins=n_bins, nonlinear=nonlinear)

    bin_col = "lifecycle_bin"
    center_col = "lifecycle_bin_center"

    def compute_percentiles_func(group):
        valid = group[value_col].dropna()
        if len(valid) == 0:
            return pd.Series({f"p{int(p)}": np.nan for p in percentiles})
        return pd.Series({
            f"p{int(p)}": valid.quantile(p / 100)
            for p in percentiles
        })

    result = df.groupby(bin_col).apply(compute_percentiles_func, include_groups=False)
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
