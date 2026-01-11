"""
Burst and Surge Detection.

This module consolidates all burst/surge detection logic including
volume surges, volatility surges, and joint volatility-volume bursts.

Definitions:
    Volume Surge:
        z_score = (volume - rolling_mean) / rolling_std
        surge = z_score > threshold (default 2.0)

        Uses 30-period rolling window with min_periods=5. Threshold of 2.0
        corresponds to approximately 2 standard deviations, capturing
        statistically unusual volume events (~2.3% of observations under
        normality assumption).

    Volatility Surge:
        z_score = (|return| - rolling_mean_volatility) / rolling_std
        surge = z_score > threshold

        Uses absolute returns (abs_return, pct_return, or log_return in
        priority order). Same z-score framework as volume surges.

    Volatility Burst (joint condition):
        |mid_return| > k * rolling_volatility AND volume > m * rolling_volume

        Joint condition (k=2.5, m=1.5) requires both high volatility AND
        elevated volume, filtering out noise-driven price moves. This
        captures information-driven events where price moves are accompanied
        by trading activity.

    Burst Intensity:
        intensity = mean((volatility - mean) / std) for burst observations

        Burst threshold = mean + 2*std. Intensity measures the average
        normalized magnitude of burst observations.

References:
    - Section 4: Volatility burst classification in regime detection
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger("microstructure.bursts")


# Data structures
# Moved from: src/microstructure/analysis.py

@dataclass
class SurgeEvent:
    """
    Container for detected surge events.

    A surge is detected when a metric's z-score exceeds the threshold.
    Magnitude is the z-score value. Baseline is the rolling mean at
    the time of detection.
    """
    contract_id: str
    timestamp: pd.Timestamp
    surge_type: str  # "volume" or "volatility"
    magnitude: float  # z-score magnitude
    baseline: float  # rolling mean at surge time
    duration_minutes: int | None = None


# Volume surge detection
# Moved from: src/microstructure/analysis.py (SurgeDetector class)

def detect_volume_surges(
    df: pd.DataFrame,
    contract_id: str,
    threshold: float = 2.0,
    lookback_window: int = 30,
) -> list[SurgeEvent]:
    """
    Detect volume surge events.

    Definition:
        Volume Surge = observations where volume z-score > threshold
        z_score = (volume - rolling_mean) / rolling_std

        Uses 30-period rolling window with min_periods=5. Zero volume
        periods are included in rolling calculations. Default threshold
        of 2.0 captures approximately 2-sigma events.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with volume data.
    contract_id : str
        Contract identifier.
    threshold : float
        Z-score threshold for surge detection (default 2.0).
    lookback_window : int
        Rolling window size (default 30).

    Returns
    -------
    list[SurgeEvent]
        Detected surge events.

    Moved from
    ----------
    src/microstructure/analysis.py:88-127 (SurgeDetector.detect_volume_surges)
    """
    if "volume" not in df.columns:
        return []

    volume = df["volume"]
    rolling_mean = volume.rolling(window=lookback_window, min_periods=5).mean()
    rolling_std = volume.rolling(window=lookback_window, min_periods=5).std()

    zscore = (volume - rolling_mean) / rolling_std.replace(0, np.nan)

    surge_mask = zscore > threshold
    surges = []

    for idx in df.index[surge_mask]:
        surges.append(SurgeEvent(
            contract_id=contract_id,
            timestamp=idx,
            surge_type="volume",
            magnitude=float(zscore.loc[idx]),
            baseline=float(rolling_mean.loc[idx]),
        ))

    return surges


def detect_volatility_surges(
    df: pd.DataFrame,
    contract_id: str,
    threshold: float = 2.0,
    lookback_window: int = 30,
) -> list[SurgeEvent]:
    """
    Detect volatility surge events.

    Definition:
        Volatility Surge = observations where |return| z-score > threshold

        Uses absolute returns, checking for abs_return, pct_return, or
        log_return columns in priority order. Same z-score framework as
        volume surges with default 2.0 threshold for ~2-sigma events.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with return data.
    contract_id : str
        Contract identifier.
    threshold : float
        Z-score threshold for surge detection.
    lookback_window : int
        Rolling window size.

    Returns
    -------
    list[SurgeEvent]
        Detected surge events.

    Moved from
    ----------
    src/microstructure/analysis.py:129-176 (SurgeDetector.detect_volatility_surges)
    """
    return_col = None
    for col in ["abs_return", "pct_return", "log_return"]:
        if col in df.columns:
            return_col = col
            break

    if return_col is None:
        return []

    returns = df[return_col].abs() if return_col != "abs_return" else df[return_col]
    rolling_mean = returns.rolling(window=lookback_window, min_periods=5).mean()
    rolling_std = returns.rolling(window=lookback_window, min_periods=5).std()

    zscore = (returns - rolling_mean) / rolling_std.replace(0, np.nan)

    surge_mask = zscore > threshold
    surges = []

    for idx in df.index[surge_mask]:
        if pd.isna(zscore.loc[idx]):
            continue
        surges.append(SurgeEvent(
            contract_id=contract_id,
            timestamp=idx,
            surge_type="volatility",
            magnitude=float(zscore.loc[idx]),
            baseline=float(rolling_mean.loc[idx]),
        ))

    return surges


def compute_surge_ratio(
    df: pd.DataFrame,
    metric: str = "volume",
    lookback_window: int = 30,
) -> pd.Series:
    """
    Compute rolling surge ratio.

    Definition:
        Surge Ratio = metric / rolling_mean

        Values > 1 indicate above-average activity. Common surge
        thresholds are 1.5x (moderate) or 2x (significant) the baseline.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    metric : str
        Column to compute ratio for.
    lookback_window : int
        Rolling window size.

    Returns
    -------
    pd.Series
        Surge ratio time series.

    Moved from
    ----------
    src/microstructure/analysis.py:178-199 (SurgeDetector.compute_surge_ratio)
    """
    if metric not in df.columns:
        return pd.Series(np.nan, index=df.index)

    values = df[metric]
    rolling_mean = values.rolling(window=lookback_window, min_periods=5).mean()
    return values / rolling_mean.replace(0, np.nan)


# Volatility burst classification
# Moved from: src/features/feature_engineering.py (MicrostructureRegimeFeatures)

def classify_volatility_burst(
    df: pd.DataFrame,
    burst_volatility_k: float = 2.5,
    burst_volume_multiplier: float = 1.5,
    rolling_window: int = 20,
) -> pd.Series:
    """
    Classify volatility burst events using joint condition.

    Definition:
        Volatility Burst = |mid_return| > k * rolling_volatility
                           AND volume > m * rolling_volume

        The joint condition (k=2.5, m=1.5) requires both elevated volatility
        and volume, filtering noise-driven price moves. Uses 20-period rolling
        window (min_periods=3 for volatility, 1 for volume). This captures
        information-driven events where significant price moves are accompanied
        by trading activity.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with pct_return and volume columns.
    burst_volatility_k : float
        Multiplier for volatility threshold (default 2.5).
    burst_volume_multiplier : float
        Multiplier for volume threshold (default 1.5).
    rolling_window : int
        Rolling window for baseline computation (default 20).

    Returns
    -------
    pd.Series
        Boolean series indicating burst observations.

    Moved from
    ----------
    src/features/feature_engineering.py:663-675 (MicrostructureRegimeFeatures.extract)
    """
    burst_mask = pd.Series(False, index=df.index)

    if "pct_return" not in df.columns:
        return burst_mask

    raw_rolling_volatility = df["pct_return"].abs().rolling(
        window=rolling_window, min_periods=3
    ).mean()

    mid_return = df["pct_return"].abs()
    volatility_condition = mid_return > (burst_volatility_k * raw_rolling_volatility)

    if "volume" in df.columns:
        raw_rolling_volume = df["volume"].rolling(
            window=rolling_window, min_periods=1
        ).mean()
        volume_condition = df["volume"] > (raw_rolling_volume * burst_volume_multiplier)
        burst_mask = volatility_condition & volume_condition
    else:
        burst_mask = volatility_condition

    return burst_mask


# Burst intensity
# Moved from: src/utils/export.py

def compute_burst_intensity(
    df: pd.DataFrame,
    burst_std_multiplier: float = 2.0,
    min_observations: int = 10,
) -> float:
    """
    Compute volatility burst intensity.

    Definition:
        Burst = observations where volatility > mean + k*std
        Intensity = mean((volatility - mean) / std) for burst observations

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with volatility data.
    burst_std_multiplier : float
        Number of standard deviations above mean to classify as burst (default 2.0).
        Higher values = stricter burst definition.
    min_observations : int
        Minimum observations required for valid calculation (default 10).

    Returns
    -------
    float
        Burst intensity metric. Returns 0.0 if no bursts, NaN if insufficient data.

    Moved from
    ----------
    src/utils/export.py:208-229 (MetricsExporter._compute_burst_intensity)
    """
    if "volatility_short" not in df.columns:
        return np.nan

    vol = df["volatility_short"].dropna()
    if len(vol) < min_observations:
        return np.nan

    mean_vol = vol.mean()
    std_vol = vol.std()

    if std_vol == 0:
        return 0.0

    burst_threshold = mean_vol + burst_std_multiplier * std_vol
    bursts = vol[vol > burst_threshold]

    if len(bursts) == 0:
        return 0.0

    return float((bursts - mean_vol).mean() / std_vol)
