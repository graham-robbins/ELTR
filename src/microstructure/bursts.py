"""
Burst and Surge Detection.

This module consolidates all burst/surge detection logic including
volume surges, volatility surges, and joint volatility-volume bursts.

Definitions:
    Volume Surge:
        z_score = (volume - rolling_mean) / rolling_std
        surge = z_score > threshold (default 2.0)
        AUTHOR MUST DEFINE FORMALLY: rolling window (30), min periods (5),
        threshold justification

    Volatility Surge:
        z_score = (|return| - rolling_mean_volatility) / rolling_std
        surge = z_score > threshold
        AUTHOR MUST DEFINE FORMALLY: return type (log, pct, abs),
        threshold value and justification

    Volatility Burst (joint condition):
        |mid_return| > k * rolling_volatility AND volume > m * rolling_volume
        AUTHOR MUST DEFINE FORMALLY: k (default 2.5), m (default 1.5),
        rolling window (default 20), joint condition rationale

    Burst Intensity:
        intensity = mean((volatility - mean) / std) for burst observations
        AUTHOR MUST DEFINE FORMALLY: aggregation method, normalization,
        threshold for burst identification (mean + 2*std)

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

    AUTHOR MUST DEFINE FORMALLY: surge event criteria,
    magnitude interpretation, baseline computation
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
        AUTHOR MUST DEFINE FORMALLY: threshold selection, window size,
        minimum periods, handling of zero/low volume periods

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
        AUTHOR MUST DEFINE FORMALLY: return type selection (abs_return, pct_return, log_return),
        threshold justification, relationship to realized volatility

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
        Indicates deviation from baseline.
        AUTHOR MUST DEFINE FORMALLY: interpretation, threshold for "surge"

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
        AUTHOR MUST DEFINE FORMALLY: k (default 2.5), m (default 1.5),
        rolling window selection, joint condition rationale,
        relationship to information arrival

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

def compute_burst_intensity(df: pd.DataFrame) -> float:
    """
    Compute volatility burst intensity.

    Definition:
        Burst = observations where volatility > mean + 2*std
        Intensity = mean((volatility - mean) / std) for burst observations
        AUTHOR MUST DEFINE FORMALLY: volatility column selection,
        threshold (2 std), intensity interpretation

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with volatility data.

    Returns
    -------
    float
        Burst intensity metric.

    Moved from
    ----------
    src/utils/export.py:208-229 (MetricsExporter._compute_burst_intensity)
    """
    if "volatility_short" not in df.columns:
        return np.nan

    vol = df["volatility_short"].dropna()
    if len(vol) < 10:
        return np.nan

    mean_vol = vol.mean()
    std_vol = vol.std()

    if std_vol == 0:
        return 0.0

    burst_threshold = mean_vol + 2 * std_vol
    bursts = vol[vol > burst_threshold]

    if len(bursts) == 0:
        return 0.0

    return float((bursts - mean_vol).mean() / std_vol)
